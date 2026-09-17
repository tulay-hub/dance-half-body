#!/usr/bin/env python3
"""
诊断脚本：检查为什么机器人不动
"""

import sys
from pathlib import Path
import torch
import numpy as np

FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
checkpoint_path = str(FRAMEWORK_ROOT / "logs/rsl_rl/lens110_amp_leg_pitch_roll/2026-07-17_16-00-14_12dof_gait_tune_stable_v2/model_49999.pt")
MODEL_PATH = str(FRAMEWORK_ROOT / "source/legged_lab/legged_lab/data/Robots/model_humanoid_lens110/mjcf/lens110_21dof.xml")

print("=" * 60)
print("🔍 Sim2Sim诊断工具")
print("=" * 60)

# 1. 检查checkpoint
print(f"\n1. 检查Checkpoint: {checkpoint_path}")

try:
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    print(f"✓ Checkpoint加载成功")
    print(f"  Keys: {list(ckpt.keys())}")

    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
        actor_keys = [k for k in state_dict.keys() if 'actor' in k.lower() and 'critic' not in k.lower()]
        print(f"  Actor参数数量: {len(actor_keys)}")
        print(f"  前5个: {actor_keys[:5]}")

        # 检查权重
        first_layer_key = actor_keys[0] if actor_keys else None
        if first_layer_key:
            weights = state_dict[first_layer_key]
            print(f"  第一层形状: {weights.shape}")
            print(f"  权重范围: [{weights.min():.3f}, {weights.max():.3f}]")
    else:
        print(f"  ⚠️ 没有找到'model_state_dict'键")
        print(f"  可用的键: {list(ckpt.keys())}")

except Exception as e:
    print(f"❌ 加载失败: {e}")

# 2. 测试策略网络
print(f"\n2. 测试策略网络")

try:
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

    # 加载权重
    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
        actor_dict = {k.replace('actor.', ''): v for k, v in state_dict.items()
                     if 'actor' in k and 'critic' not in k}

        print(f"  提取的actor参数: {len(actor_dict)}")

        # 尝试加载
        missing, unexpected = policy.load_state_dict(actor_dict, strict=False)
        print(f"  缺失的键: {len(missing)}")
        print(f"  多余的键: {len(unexpected)}")

        if missing:
            print(f"  ⚠️ 缺失: {missing}")
        if unexpected:
            print(f"  ⚠️ 多余: {unexpected}")

    # 测试推理
    print(f"\n3. 测试推理")
    policy.eval()
    dummy_obs = torch.randn(1, 45)

    with torch.no_grad():
        action = policy(dummy_obs)

    print(f"  输入形状: {dummy_obs.shape}")
    print(f"  输出形状: {action.shape}")
    print(f"  动作范围: [{action.min():.3f}, {action.max():.3f}]")
    print(f"  动作均值: {action.mean():.3f}")
    print(f"  动作标准差: {action.std():.3f}")

    if action.abs().max() < 0.01:
        print(f"  ⚠️ 警告: 动作幅度过小！")
    else:
        print(f"  ✓ 动作幅度正常")

except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()

# 4. 检查Mujoco模型
print(f"\n4. 检查Mujoco模型")
try:
    import mujoco
    model_path = MODEL_PATH

    model = mujoco.MjModel.from_xml_path(model_path)
    print(f"  ✓ Mujoco模型加载成功")
    print(f"  自由度: {model.nv}")
    print(f"  执行器数量: {model.nu}")
    print(f"  关节数量: {model.njnt}")

    # 检查关节名称
    print(f"\n  关节列表:")
    for i in range(min(model.njnt, 15)):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        print(f"    {i}: {joint_name}")

except Exception as e:
    print(f"❌ Mujoco模型加载失败: {e}")

print("\n" + "=" * 60)
print("诊断完成")
print("=" * 60)
