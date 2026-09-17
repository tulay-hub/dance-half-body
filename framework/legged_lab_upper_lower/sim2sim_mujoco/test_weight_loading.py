#!/usr/bin/env python3
"""
测试权重加载是否正确
"""

import torch
from pathlib import Path
from torch import nn

FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
checkpoint_path = str(FRAMEWORK_ROOT / "logs/rsl_rl/lens110_amp_leg_pitch_roll/2026-07-17_16-00-14_12dof_gait_tune_stable_v2/model_49999.pt")

print("=" * 60)
print("测试权重加载")
print("=" * 60)

# 加载checkpoint
ckpt = torch.load(checkpoint_path, map_location='cpu')
state_dict = ckpt['model_state_dict']

# 打印所有actor键
print("\n原始checkpoint中的actor键:")
for k in sorted(state_dict.keys()):
    if 'actor' in k and 'critic' not in k:
        print(f"  {k}: {state_dict[k].shape}")

# 创建网络
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

print("\n网络期望的键:")
for k, v in policy.state_dict().items():
    print(f"  {k}: {v.shape}")

# 方法1: 直接加载（checkpoint的键已经是actor.x.weight格式）
print("\n方法1: 直接加载actor权重")
actor_state = {k: v for k, v in state_dict.items() if 'actor' in k and 'critic' not in k}
result = policy.load_state_dict(actor_state, strict=False)
print(f"  缺失: {result.missing_keys}")
print(f"  多余: {result.unexpected_keys}")

# 测试推理
print("\n测试推理:")
policy.eval()
dummy_obs = torch.randn(1, 45) * 0.1  # 小的输入

with torch.no_grad():
    action = policy(dummy_obs)

print(f"  输出: {action[0][:6].numpy()}")
print(f"  范围: [{action.min():.3f}, {action.max():.3f}]")
print(f"  均值: {action.mean():.3f}")
print(f"  标准差: {action.std():.3f}")

# 检查第一层权重
first_weight = policy.actor[0].weight
print(f"\n第一层权重统计:")
print(f"  形状: {first_weight.shape}")
print(f"  范围: [{first_weight.min():.3f}, {first_weight.max():.3f}]")
print(f"  均值: {first_weight.mean():.3f}")
print(f"  标准差: {first_weight.std():.3f}")

# 对比checkpoint中的权重
ckpt_weight = state_dict['actor.0.weight']
print(f"\nCheckpoint第一层权重:")
print(f"  范围: [{ckpt_weight.min():.3f}, {ckpt_weight.max():.3f}]")
print(f"  均值: {ckpt_weight.mean():.3f}")

if torch.allclose(first_weight, ckpt_weight):
    print("\n✅ 权重加载成功！")
else:
    print("\n❌ 权重加载失败！")

print("=" * 60)
