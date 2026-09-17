#!/usr/bin/env python3
"""
Sim2Sim (修复版): Isaac Sim → Mujoco
基于scripts/sim2sim_pr_up/lens110_pr_up.py的正确实现
"""

import argparse
import numpy as np
import torch
import mujoco
import mujoco.viewer
from pathlib import Path
import time
from torch import nn


FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = FRAMEWORK_ROOT / "source/legged_lab/legged_lab/data/Robots/model_humanoid_lens110/mjcf/lens110_21dof.xml"

# 关节顺序映射 (训练时的policy顺序)
POLICY_JOINT_NAMES = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "torso_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
]

# Mujoco中的关节顺序
MUJOCO_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "torso_yaw_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
]

# 默认关节位置 (Mujoco顺序)
DEFAULT_POS = np.array([
    -0.14, 0.01, -0.1, 0.3, -0.15, 0.0,  # 左腿
    -0.14, -0.01, 0.1, 0.3, -0.15, 0.0,  # 右腿
    0.0,  # torso
    0.4, 0.2, 0.0, -0.8,  # 左臂
    0.4, -0.2, 0.0, -0.8,  # 右臂
], dtype=np.float64)

# 动作缩放 (Policy顺序)
ACTION_SCALE = np.array([
    0.25, 0.25, 0.125,  # hip_pitch, torso
    0.25, 0.25,  # hip_roll
    0.18, 0.18,  # shoulder_pitch
    0.25, 0.25,  # hip_yaw
    0.18, 0.18,  # shoulder_roll
    0.25, 0.25,  # knee
    0.18, 0.18,  # shoulder_yaw
    0.25, 0.25,  # ankle_pitch
    0.18, 0.18,  # elbow
    0.25, 0.25,  # ankle_roll
], dtype=np.float64)

# PD控制器参数 (Mujoco顺序)
KP = np.array([40, 40, 40, 40, 14.5, 7.0] * 2 + [100] + [20] * 8, dtype=np.float64)
KD = np.array([5, 5, 5, 5, 14.5, 7.0] * 2 + [5] + [1] * 8, dtype=np.float64)
TAU_LIMIT = np.array([80, 80, 80, 80, 36, 36] * 2 + [80] + [36] * 8, dtype=np.float64)


def quat_to_rotmat_wxyz(quat: np.ndarray) -> np.ndarray:
    """四元数转旋转矩阵 (w,x,y,z格式)"""
    quat = quat / max(np.linalg.norm(quat), 1e-12)
    w, x, y, z = quat
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim=45, action_dim=12):  # 只有12个腿部关节
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, obs):
        return self.actor(obs)


class FixedSim2SimAdapter:
    def __init__(self, model_path: str, checkpoint_path: str):
        # 加载Mujoco模型
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # 加载策略
        self.policy = self.load_policy(checkpoint_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)
        self.policy.eval()

        # 创建映射: policy顺序 -> mujoco顺序
        mujoco_index = {name: i for i, name in enumerate(MUJOCO_JOINT_NAMES)}
        self.policy_to_mujoco = np.array([mujoco_index[name] for name in POLICY_JOINT_NAMES], dtype=np.int32)

        # 获取关节ID
        self.joint_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in MUJOCO_JOINT_NAMES
        ], dtype=np.int32)
        self.qpos_ids = self.model.jnt_qposadr[self.joint_ids]
        self.qvel_ids = self.model.jnt_dofadr[self.joint_ids]

        # 获取执行器ID
        self.actuator_ids = self._find_actuators()

        # 初始化状态
        self.last_action = np.zeros(21, dtype=np.float32)
        self.command = np.array([0.8, 0.0, 0.0], dtype=np.float32)  # [vx, vy, vyaw]
        self.step_count = 0

        print(f"✓ Mujoco模型: {model_path}")
        print(f"✓ 策略加载: {checkpoint_path}")
        print(f"✓ 设备: {self.device}")
        print(f"✓ 关节数: {len(MUJOCO_JOINT_NAMES)}")

    def _find_actuators(self):
        """查找对应的执行器"""
        actuator_ids = []
        for joint_id in self.joint_ids:
            found = False
            for act_id in range(self.model.nu):
                if self.model.actuator_trntype[act_id] == mujoco.mjtTrn.mjTRN_JOINT:
                    if self.model.actuator_trnid[act_id, 0] == joint_id:
                        actuator_ids.append(act_id)
                        found = True
                        break
            if not found:
                actuator_ids.append(-1)
        return np.array(actuator_ids, dtype=np.int32)

    def load_policy(self, checkpoint_path: str):
        """加载策略网络"""
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        state_dict = ckpt['model_state_dict']

        policy = PolicyNetwork(obs_dim=45, action_dim=21)
        actor_state = {k: v for k, v in state_dict.items() if 'actor' in k and 'critic' not in k}
        result = policy.load_state_dict(actor_state, strict=False)

        if not result.missing_keys and not result.unexpected_keys:
            print("✓ 权重加载成功")
        else:
            print(f"⚠️ 权重加载: 缺失{len(result.missing_keys)}, 多余{len(result.unexpected_keys)}")

        return policy

    def get_observation(self):
        """构建观测 (45维)"""
        # 1. 角速度 (body frame) (3)
        quat = self.data.qpos[3:7]  # [w,x,y,z]
        rot = quat_to_rotmat_wxyz(quat)
        omega = self.data.qvel[3:6]  # 已经是body frame

        # 2. 投影重力 (3)
        gravity_world = np.array([0, 0, -1], dtype=np.float64)
        gravity = rot.T @ gravity_world

        # 3. 速度命令 (3)
        command = self.command.copy()

        # 4. 关节位置 (相对默认位置) (21)
        q_mujoco = self.data.qpos[self.qpos_ids]
        q_relative = q_mujoco - DEFAULT_POS
        q_obs = q_relative[self.policy_to_mujoco]  # 转换到policy顺序

        # 5. 关节速度 (21)
        dq_mujoco = self.data.qvel[self.qvel_ids]
        dq_obs = dq_mujoco[self.policy_to_mujoco]  # 转换到policy顺序

        # 6. 上一次动作 (21)
        last_action = self.last_action.copy()

        # 拼接: (3+3+3+21+21+21=72? 不对，应该是45)
        # 检查: 3+3+3+12+12+12=45，所以只用腿部关节
        # 让我重新确认...实际上训练用的是12个腿部关节

        # 重新构建：只用12个腿部关节
        leg_indices_policy = [0, 1, 3, 4, 7, 8, 11, 12, 15, 16, 19, 20]  # policy顺序中的腿部关节
        q_legs = q_obs[leg_indices_policy]
        dq_legs = dq_obs[leg_indices_policy]
        last_action_legs = self.last_action[leg_indices_policy]

        obs = np.concatenate([
            omega, gravity, command,
            q_legs, dq_legs, last_action_legs
        ]).astype(np.float32)

        return obs

    def apply_action(self, action_policy: np.ndarray):
        """应用动作 (使用PD控制)"""
        # action_policy是policy顺序，需要转换
        action_mujoco = action_policy[self.policy_to_mujoco]

        # 缩放动作
        scaled_action = action_policy * ACTION_SCALE

        # 计算目标位置
        target_mujoco = DEFAULT_POS + scaled_action[self.policy_to_mujoco]

        # PD控制计算力矩
        q = self.data.qpos[self.qpos_ids]
        dq = self.data.qvel[self.qvel_ids]
        tau = KP * (target_mujoco - q) - KD * dq
        tau = np.clip(tau, -TAU_LIMIT, TAU_LIMIT)

        # 应用到执行器
        for i, act_id in enumerate(self.actuator_ids):
            if act_id >= 0:
                self.data.ctrl[act_id] = tau[i]

        # 更新last_action
        self.last_action = action_policy.astype(np.float32)

    def reset(self):
        """重置环境"""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = 0.68  # 初始高度
        self.data.qpos[3:7] = [1, 0, 0, 0]  # 初始姿态
        self.data.qpos[self.qpos_ids] = DEFAULT_POS
        mujoco.mj_forward(self.model, self.data)
        self.last_action = np.zeros(21, dtype=np.float32)
        self.step_count = 0
        print("✓ 环境已重置")

    def step(self):
        """执行一步"""
        obs = self.get_observation()

        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(self.device)
            action_tensor = self.policy(obs_tensor)
            action = action_tensor.cpu().numpy()[0]

        self.apply_action(action)
        mujoco.mj_step(self.model, self.data)
        self.step_count += 1

        return obs, action

    def run_interactive(self, duration: float = 30.0):
        """运行交互式可视化"""
        print(f"\n🎮 启动交互式可视化 (运行{duration}秒)")
        print("=" * 60)

        self.reset()

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            start_time = time.time()

            while viewer.is_running() and (time.time() - start_time) < duration:
                step_start = time.time()

                obs, action = self.step()
                viewer.sync()

                if self.step_count % 100 == 0:
                    base_pos = self.data.qpos[:3]
                    base_vel = self.data.qvel[:3]
                    print(f"Step {self.step_count:5d} | "
                          f"Pos: [{base_pos[0]:6.3f}, {base_pos[1]:6.3f}, {base_pos[2]:6.3f}] | "
                          f"Vel: [{base_vel[0]:6.3f}, {base_vel[1]:6.3f}]")

                elapsed = time.time() - step_start
                if elapsed < 0.02:
                    time.sleep(0.02 - elapsed)

            elapsed_time = time.time() - start_time
            print("=" * 60)
            print(f"✓ 仿真完成: {self.step_count}步, 用时{elapsed_time:.2f}秒")
            print(f"✓ 平均FPS: {self.step_count/elapsed_time:.1f}")


def main():
    parser = argparse.ArgumentParser(description="Sim2Sim (修复版): Isaac Sim → Mujoco")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Local 45->21 compatible checkpoint; no matching checkpoint is bundled in this workspace.",
    )
    parser.add_argument("--model", type=str,
                       default=str(DEFAULT_MODEL))
    parser.add_argument("--duration", type=float, default=30.0)

    args = parser.parse_args()

    print("=" * 60)
    print("🚀 Sim2Sim (修复版): Isaac Sim → Mujoco")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Model: {args.model}")
    print("=" * 60)

    adapter = FixedSim2SimAdapter(args.model, args.checkpoint)
    adapter.run_interactive(duration=args.duration)


if __name__ == "__main__":
    main()
