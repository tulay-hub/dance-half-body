# 跳舞半身（upper/lower）奖励结构

任务层使用速度跟踪、站高/姿态、步态节奏、脚距、滑步、关节限位、力矩、动作平滑和自碰撞奖励；AMP
discriminator 另提供 style reward。奖励实现基于：
`framework/legged_lab_upper_lower/source/legged_lab/legged_lab/tasks/locomotion/amp/config/lens110/`。

本项目的核心差异是 policy-facing ankle state/action 为 upper/lower，经过 polynomial、weighted MLP、
XML tendon 或直接 MJCF plant 后才落到物理 pitch/roll/执行器。不要跨变体混用奖励、模型和部署配置。

逐项权重见 [`docs/REWARD_FRAMEWORKS.md`](../../../docs/REWARD_FRAMEWORKS.md)；映射链路见
[`docs/INTERFACE_CONTRACTS.md`](../../../docs/INTERFACE_CONTRACTS.md)。

## 训练权重如何理解 / Interpreting training weights

本文按本仓库当前代码说明训练机制；已有策略的复现参数以对应 run 的 `params/env.yaml`、`params/agent.yaml` 和部署配置为准。奖励混合系数、逐项环境奖励权重、优化器 loss 系数、专家样本比例以及课程采样范围是不同概念。

混合系数可以写成 85%/15% 这样的配置比例，但不能代表训练过程中实际累计奖励贡献；单项 reward 的数值范围、门控、控制步长和出现频率都不同。需要实际贡献占比时，应统计同一 run 中每项加权回报，而不是把配置权重归一化成百分比。

Configuration mixing coefficients are not measured reward contributions. Environment weights, optimizer coefficients, expert sampling and curriculum schedules describe different parts of training. Reproduce a saved policy with its own run snapshots.

## AMP 专家先验、混合比例与实际时序

本任务用 PPOAMP 学习任务控制，并用专家 motion 的 LSGAN 判别器提供动作先验。实际奖励不是简单的 task+style 相加：

```text
phi(D) = max(0, 1 - 0.25*(D-1)^2)
r_style = dt_control * 5.0 * phi(D)
r_PPO = 0.4*r_task + 0.6*r_style
dt_control = 0.005 * 4 = 0.02 s
r_PPO = 0.4*r_task + 0.06*phi(D)
```

40%/60% 是任务/风格混合系数。AMP 的 5.0 scale 还要乘 0.02 s 控制步长，不能与 GetUp 的 0.1 scale 跨框架直接比较。当前物理 `200 Hz`、策略 `50 Hz`；actor/critic 历史为 1/3 帧，判别器与 demo 均由 `AMP_NUM_STEPS=3` 覆盖基类 10 帧默认值。

PPO 配置：value=1.0、entropy=0.01、clip=0.2、gamma=0.99、lambda=0.95，16 rollout steps、5 epochs、8 mini-batches；策略初始 LR=1e-4（adaptive），判别器 LR=1e-4、gradient penalty scale=10。upper/lower runner 将 `symmetry_cfg=None`，因此该变体没有启用基类的 mirror loss；环境中的 arm/hip 对称 reward 仍按环境配置计算。

源码：`framework/legged_lab_upper_lower/source/legged_lab/legged_lab/tasks/locomotion/amp/amp_env_cfg.py`，`config/lens110/lens110_amp_env_cfg.py` 及 `config/lens110/agents/rsl_rl_ppo_cfg.py`（config 相对同一 amp 目录），以及 `framework/legged_lab_upper_lower/rsl_rl/rsl_rl/modules/amp.py`。Polynomial/MLP/XML tendon/direct MJCF 各自定义状态与动作映射，复现必须匹配注册的 runner 和 plant。

English: The actual blend is `0.4*task + 0.6*(0.02*5*phi(D))`. Physics/control are 200/50 Hz. Discriminator/demo history is 3 frames, actor/critic history is 1/3. Upper/lower runner disables algorithmic symmetry loss, independently of environmental symmetry rewards. PPO value/entropy coefficients are 1.0/0.01; policy and discriminator initial learning rates are 1e-4.
