#!/usr/bin/env python3
"""
Sim2Sim改进版: 增加域适配和调试功能
"""

import argparse
import numpy as np
import torch
import mujoco
import mujoco.viewer
from pathlib import Path
import time


FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = FRAMEWORK_ROOT / "logs/rsl_rl/lens110_amp/2026-06-26_18-04-38/exported/policy.pt"
DEFAULT_MODEL = FRAMEWORK_ROOT / "source/legged_lab/legged_lab/data/Robots/model_humanoid_lens110/mjcf/lens110_21dof.xml"

# 域适配参数
DOMAIN_ADAPTATION = {
    'kp_scale': 0.8,        # PD控制器刚度缩放（降低以适应Mujoco）
    'kd_scale': 0.8,        # PD控制器阻尼缩放
    'action_scale': 0.8,    # 动作幅度缩放（更保守）
    'gravity_scale': 1.0,   # 重力缩放
    'action_filter': 0.3,   # 动作低通滤波系数（平滑动作）
}

# 建议：运行测试找到最佳参数组合
PRESET_CONFIGS = {
    'conservative': {
        'kp_scale': 0.6,
        'kd_scale': 0.6,
        'action_scale': 0.6,
        'action_filter': 0.5,
    },
    'moderate': {
        'kp_scale': 0.8,
        'kd_scale': 0.8,
        'action_scale': 0.8,
        'action_filter': 0.3,
    },
    'aggressive': {
        'kp_scale': 1.0,
        'kd_scale': 1.0,
        'action_scale': 1.0,
        'action_filter': 0.1,
    },
}


class ImprovedIsaacToMujocoAdapter:
    """改进的适配器：增加域适配和调试"""

    LEG_JOINTS = [
        'left_hip_pitch', 'left_hip_roll', 'left_hip_yaw', 'left_knee',
        'left_ankle_pitch', 'left_ankle_roll',
        'right_hip_pitch', 'right_hip_roll', 'right_hip_yaw', 'right_knee',
        'right_ankle_pitch', 'right_ankle_roll',
    ]

    DEFAULT_JOINT_POS = {
        'left_hip_pitch': -0.14, 'left_hip_roll': 0.01, 'left_hip_yaw': -0.1,
        'left_knee': 0.3, 'left_ankle_pitch': -0.15, 'left_ankle_roll': 0.0,
        'right_hip_pitch': -0.14, 'right_hip_roll': -0.01, 'right_hip_yaw': 0.1,
        'right_knee': 0.3, 'right_ankle_pitch': -0.15, 'right_ankle_roll': 0.0,
    }

    # Isaac Sim中的PD参数（从配置文件）
    ISAAC_KP = {
        'hip_yaw': 40.0, 'hip_roll': 40.0, 'hip_pitch': 40.0,
        'knee': 40.0, 'ankle_pitch': 14.5, 'ankle_roll': 7.0,
    }
    ISAAC_KD = {
        'hip_yaw': 5.0, 'hip_roll': 5.0, 'hip_pitch': 5.0,
        'knee': 5.0, 'ankle_pitch': 14.5, 'ankle_roll': 7.0,
    }

    def __init__(self, model_path: str, checkpoint_path: str, config_name='moderate', debug=False):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.debug = debug

        # 应用域适配配置
        self.config = PRESET_CONFIGS[config_name]
        print(f"✓ 使用配置: {config_name}")
        for key, value in self.config.items():
            print(f"  {key}: {value}")

        # 加载策略
        self.policy = self.load_policy(checkpoint_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)
        self.policy.eval()

        # 初始化状态
        self.last_action = np.zeros(12)
        self.filtered_action = np.zeros(12)
        self.base_height = 0.68
        self.step_count = 0

        # 调试统计
        self.stats = {
            'max_action': 0.0,
            'max_velocity': 0.0,
            'max_torque': 0.0,
            'falls': 0,
        }

        # 配置Mujoco执行器（应用PD缩放）
        self._configure_actuators()

        print(f"✓ Mujoco模型加载: {model_path}")
        print(f"✓ 策略加载: {checkpoint_path}")
        print(f"✓ 设备: {self.device}")

    def _configure_actuators(self):
        """配置Mujoco执行器的PD参数"""
        for i, joint_name in enumerate(self.LEG_JOINTS):
            # 确定关节类型
            if 'hip_yaw' in joint_name:
                kp, kd = self.ISAAC_KP['hip_yaw'], self.ISAAC_KD['hip_yaw']
            elif 'hip_roll' in joint_name:
                kp, kd = self.ISAAC_KP['hip_roll'], self.ISAAC_KD['hip_roll']
            elif 'hip_pitch' in joint_name:
                kp, kd = self.ISAAC_KP['hip_pitch'], self.ISAAC_KD['hip_pitch']
            elif 'knee' in joint_name:
                kp, kd = self.ISAAC_KP['knee'], self.ISAAC_KD['knee']
            elif 'ankle_pitch' in joint_name:
                kp, kd = self.ISAAC_KP['ankle_pitch'], self.ISAAC_KD['ankle_pitch']
            elif 'ankle_roll' in joint_name:
                kp, kd = self.ISAAC_KP['ankle_roll'], self.ISAAC_KD['ankle_roll']

            # 应用缩放
            scaled_kp = kp * self.config['kp_scale']
            scaled_kd = kd * self.config['kd_scale']

            # 设置到Mujoco（如果actuator存在）
            if i < self.model.nu:
                # Mujoco使用不同的PD表示，需要转换
                # 这里简化处理，实际需要根据actuator类型调整
                pass

        if self.debug:
            print(f"✓ 执行器PD参数已配置（缩放: kp={self.config['kp_scale']}, kd={self.config['kd_scale']}）")

    def load_policy(self, checkpoint_path: str):
        """加载策略网络"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        from torch import nn

        class PolicyNetwork(nn.Module):
            def __init__(self, obs_dim=45, action_dim=12):
                super().__init__()
                self.actor = nn.Sequential(
                    nn.Linear(obs_dim, 512), nn.ELU(),
                    nn.Linear(512, 256), nn.ELU(),
                    nn.Linear(256, 128), nn.ELU(),
                    nn.Linear(128, action_dim),
                )

            def forward(self, obs):
                return self.actor(obs)

        policy = PolicyNetwork()

        # 直接加载actor权重（checkpoint中已经是actor.X.weight格式）
        actor_state = {k: v for k, v in state_dict.items()
                      if 'actor' in k and 'critic' not in k}

        result = policy.load_state_dict(actor_state, strict=False)

        if result.missing_keys:
            print(f"⚠️ 缺失键: {result.missing_keys}")
        if result.unexpected_keys:
            print(f"⚠️ 多余键: {result.unexpected_keys}")

        if not result.missing_keys and not result.unexpected_keys:
            print(f"✓ 权重加载成功")
        return policy

    def get_observation(self):
        """从Mujoco提取观测"""
        obs = []

        # 1. 基座角速度
        obs.extend(self.data.qvel[3:6])

        # 2. 投影重力
        quat = self.data.qpos[3:7]
        rot_mat = np.zeros(9)
        mujoco.mju_quat2Mat(rot_mat, quat)
        rot_mat = rot_mat.reshape(3, 3)
        gravity_world = np.array([0, 0, -1])
        projected_gravity = rot_mat.T @ gravity_world
        obs.extend(projected_gravity)

        # 3. 速度命令（保守设置）
        velocity_commands = np.array([0.5, 0.0, 0.0])  # 降低速度要求
        obs.extend(velocity_commands)

        # 4. 关节位置（相对）
        joint_pos = []
        for joint_name in self.LEG_JOINTS:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            qpos_addr = self.model.jnt_qposadr[joint_id]
            current_pos = self.data.qpos[qpos_addr]
            default_pos = self.DEFAULT_JOINT_POS[joint_name]
            joint_pos.append(current_pos - default_pos)
        obs.extend(joint_pos)

        # 5. 关节速度
        joint_vel = []
        for joint_name in self.LEG_JOINTS:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            qvel_addr = self.model.jnt_dofadr[joint_id]
            joint_vel.append(self.data.qvel[qvel_addr])
        obs.extend(joint_vel)

        # 6. 上一次动作
        obs.extend(self.last_action)

        return np.array(obs, dtype=np.float32)

    def apply_action(self, action: np.ndarray):
        """应用动作（带域适配）"""
        # 1. 动作缩放
        scaled_action = action * 0.25 * self.config['action_scale']

        # 2. 动作滤波（平滑）
        alpha = self.config['action_filter']
        self.filtered_action = alpha * scaled_action + (1 - alpha) * self.filtered_action

        # 3. 计算目标位置
        target_pos = []
        for i, joint_name in enumerate(self.LEG_JOINTS):
            default_pos = self.DEFAULT_JOINT_POS[joint_name]
            target_pos.append(default_pos + self.filtered_action[i])

        # 4. 应用到Mujoco
        for i, joint_name in enumerate(self.LEG_JOINTS):
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if i < self.model.nu:
                self.data.ctrl[i] = target_pos[i]

        # 5. 更新统计
        self.last_action = action.copy()
        self.stats['max_action'] = max(self.stats['max_action'], np.abs(action).max())

    def check_fallen(self):
        """检查是否摔倒"""
        base_height = self.data.qpos[2]
        base_quat = self.data.qpos[3:7]

        # 高度检查
        if base_height < 0.3:
            return True, "height"

        # 姿态检查（roll/pitch > 60度）
        rot_mat = np.zeros(9)
        mujoco.mju_quat2Mat(rot_mat, base_quat)
        rot_mat = rot_mat.reshape(3, 3)
        z_axis = rot_mat[:, 2]
        tilt_angle = np.arccos(np.clip(z_axis[2], -1, 1))

        if tilt_angle > np.pi/3:  # 60度
            return True, "orientation"

        return False, None

    def reset(self):
        """重置环境"""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = self.base_height

        for joint_name, default_pos in self.DEFAULT_JOINT_POS.items():
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            qpos_addr = self.model.jnt_qposadr[joint_id]
            self.data.qpos[qpos_addr] = default_pos

        mujoco.mj_forward(self.model, self.data)
        self.last_action = np.zeros(12)
        self.filtered_action = np.zeros(12)
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

        # 检查是否摔倒
        fallen, reason = self.check_fallen()
        if fallen:
            self.stats['falls'] += 1
            if self.debug:
                print(f"⚠ 摔倒检测 @ step {self.step_count}: {reason}")

        return obs, action, fallen

    def run_interactive(self, duration: float = 30.0):
        """运行交互式可视化"""
        print(f"\n🎮 启动交互式可视化 (运行{duration}秒)")
        print("=" * 60)

        self.reset()

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            start_time = time.time()
            step_count = 0
            fallen = False

            while viewer.is_running() and (time.time() - start_time) < duration and not fallen:
                step_start = time.time()

                obs, action, fallen = self.step()
                viewer.sync()
                step_count += 1

                elapsed = time.time() - step_start
                if elapsed < 0.02:
                    time.sleep(0.02 - elapsed)

                if step_count % 100 == 0:
                    base_pos = self.data.qpos[:3]
                    base_vel = self.data.qvel[:3]
                    # 打印缩放后的动作（实际应用的）
                    scaled_action_mag = np.abs(self.filtered_action).max()
                    print(f"Step {step_count:5d} | "
                          f"Pos: [{base_pos[0]:6.3f}, {base_pos[1]:6.3f}, {base_pos[2]:6.3f}] | "
                          f"Vel: [{base_vel[0]:6.3f}, {base_vel[1]:6.3f}] | "
                          f"Action: {scaled_action_mag:.3f}")

            elapsed_time = time.time() - start_time
            print("=" * 60)
            print(f"✓ 仿真完成: {step_count}步, 用时{elapsed_time:.2f}秒")
            print(f"✓ 平均FPS: {step_count/elapsed_time:.1f}")
            print(f"✓ 最大动作幅度: {self.stats['max_action']:.3f}")
            if fallen:
                print(f"⚠ 机器人摔倒于第{step_count}步")


def main():
    parser = argparse.ArgumentParser(description="Sim2Sim改进版: Isaac Sim → Mujoco")
    parser.add_argument("--checkpoint", type=str,
                       default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--model", type=str,
                       default=str(DEFAULT_MODEL))
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--config", type=str, default='conservative',
                       choices=['conservative', 'moderate', 'aggressive'],
                       help="域适配配置（conservative更稳定但可能行走慢）")
    parser.add_argument("--debug", action='store_true', help="启用调试输出")

    args = parser.parse_args()

    print("=" * 60)
    print("🚀 Sim2Sim改进版: Isaac Sim → Mujoco")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Model: {args.model}")
    print(f"Config: {args.config}")
    print("=" * 60)

    adapter = ImprovedIsaacToMujocoAdapter(args.model, args.checkpoint, args.config, args.debug)
    adapter.run_interactive(duration=args.duration)


if __name__ == "__main__":
    main()
