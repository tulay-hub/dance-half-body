#!/usr/bin/env python3
"""
Sim2Sim简化版：基于参考代码的正确实现
只控制12个腿部关节
"""

import argparse
import numpy as np
import torch
import mujoco
import mujoco.viewer
import time
from pathlib import Path
from torch import nn


FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = FRAMEWORK_ROOT / "logs/rsl_rl/lens110_amp_leg_pitch_roll/2026-07-17_16-00-14_12dof_gait_tune_stable_v2/model_49999.pt"
DEFAULT_MODEL = FRAMEWORK_ROOT / "source/legged_lab/legged_lab/data/Robots/model_humanoid_lens110/mjcf/lens110_21dof.xml"

# 12个腿部关节 (Mujoco顺序)
LEG_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
]

# 默认位置
DEFAULT_POS = np.array([
    -0.14, 0.01, -0.1, 0.3, -0.15, 0.0,  # 左腿
    -0.14, -0.01, 0.1, 0.3, -0.15, 0.0,  # 右腿
], dtype=np.float64)

# 动作缩放
ACTION_SCALE = np.array([0.25] * 12, dtype=np.float64)

# PD参数
KP = np.array([40, 40, 40, 40, 14.5, 7.0] * 2, dtype=np.float64)
KD = np.array([5, 5, 5, 5, 14.5, 7.0] * 2, dtype=np.float64)
TAU_LIMIT = np.array([80, 80, 80, 80, 36, 36] * 2, dtype=np.float64)


def quat_to_rotmat(quat):
    """四元数转旋转矩阵"""
    quat = quat / max(np.linalg.norm(quat), 1e-12)
    w, x, y, z = quat
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(45, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 12),
        )

    def forward(self, obs):
        return self.actor(obs)


class SimpleSim2Sim:
    def __init__(self, model_path, checkpoint_path):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # 加载策略
        self.policy = PolicyNetwork()
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        actor_state = {k: v for k, v in ckpt['model_state_dict'].items()
                      if 'actor' in k and 'critic' not in k}
        self.policy.load_state_dict(actor_state, strict=False)
        self.policy.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)

        # 获取关节ID
        self.joint_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in LEG_JOINT_NAMES
        ], dtype=np.int32)
        self.qpos_ids = self.model.jnt_qposadr[self.joint_ids]
        self.qvel_ids = self.model.jnt_dofadr[self.joint_ids]

        # 找到执行器
        self.actuator_ids = []
        for joint_id in self.joint_ids:
            for act_id in range(self.model.nu):
                if self.model.actuator_trnid[act_id, 0] == joint_id:
                    self.actuator_ids.append(act_id)
                    break
        self.actuator_ids = np.array(self.actuator_ids, dtype=np.int32)

        self.last_action = np.zeros(12, dtype=np.float32)
        self.command = np.array([0.8, 0.0, 0.0], dtype=np.float32)

        print(f"✓ 模型: {model_path}")
        print(f"✓ 策略: {checkpoint_path}")
        print(f"✓ 设备: {self.device}")

    def get_observation(self):
        """构建45维观测"""
        # 角速度 (3)
        omega = self.data.qvel[3:6]

        # 投影重力 (3)
        quat = self.data.qpos[3:7]
        rot = quat_to_rotmat(quat)
        gravity = rot.T @ np.array([0, 0, -1], dtype=np.float64)

        # 速度命令 (3)
        command = self.command.copy()

        # 关节位置相对值 (12)
        q = self.data.qpos[self.qpos_ids]
        q_rel = q - DEFAULT_POS

        # 关节速度 (12)
        dq = self.data.qvel[self.qvel_ids]

        # 上一次动作 (12)
        last_act = self.last_action.copy()

        # 拼接
        obs = np.concatenate([omega, gravity, command, q_rel, dq, last_act]).astype(np.float32)
        return obs

    def apply_action(self, action):
        """应用动作（PD控制）"""
        # 缩放
        scaled = action * ACTION_SCALE

        # 目标位置
        target = DEFAULT_POS + scaled

        # PD控制
        q = self.data.qpos[self.qpos_ids]
        dq = self.data.qvel[self.qvel_ids]
        tau = KP * (target - q) - KD * dq
        tau = np.clip(tau, -TAU_LIMIT, TAU_LIMIT)

        # 应用到执行器
        for i, act_id in enumerate(self.actuator_ids):
            if act_id < self.model.nu:
                self.data.ctrl[act_id] = tau[i]

        self.last_action = action.astype(np.float32)

    def reset(self):
        """重置"""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = 0.68
        self.data.qpos[3:7] = [1, 0, 0, 0]
        self.data.qpos[self.qpos_ids] = DEFAULT_POS
        mujoco.mj_forward(self.model, self.data)
        self.last_action = np.zeros(12, dtype=np.float32)
        print("✓ 环境已重置")

    def run(self, duration=30.0):
        """运行"""
        print(f"\n🎮 运行{duration}秒")
        print("=" * 60)

        self.reset()

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            start_time = time.time()
            step = 0

            while viewer.is_running() and (time.time() - start_time) < duration:
                # 获取观测
                obs = self.get_observation()

                # 策略推理
                with torch.no_grad():
                    obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
                    action_t = self.policy(obs_t)
                    action = action_t.cpu().numpy()[0]

                # 应用动作
                self.apply_action(action)

                # 仿真
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
                step += 1

                # 打印
                if step % 100 == 0:
                    pos = self.data.qpos[:3]
                    vel = self.data.qvel[:3]
                    print(f"Step {step:5d} | Pos: [{pos[0]:6.3f}, {pos[1]:6.3f}, {pos[2]:6.3f}] | "
                          f"Vel: [{vel[0]:6.3f}, {vel[1]:6.3f}]")

                # 控制帧率
                time.sleep(0.002)

            print("=" * 60)
            print(f"✓ 完成: {step}步, {time.time()-start_time:.1f}秒")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--duration", type=float, default=30.0)
    args = parser.parse_args()

    print("=" * 60)
    print("🚀 Sim2Sim简化版")
    print("=" * 60)

    sim = SimpleSim2Sim(args.model, args.checkpoint)
    sim.run(args.duration)


if __name__ == "__main__":
    main()
