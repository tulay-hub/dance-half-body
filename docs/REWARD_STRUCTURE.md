# 跳舞半身（upper/lower）奖励结构

任务层使用速度跟踪、站高/姿态、步态节奏、脚距、滑步、关节限位、力矩、动作平滑和自碰撞奖励；AMP
discriminator 另提供 style reward。奖励实现基于：
`framework/legged_lab_upper_lower/source/legged_lab/legged_lab/tasks/locomotion/amp/config/lens110/`。

本项目的核心差异是 policy-facing ankle state/action 为 upper/lower，经过 polynomial、weighted MLP、
XML tendon 或直接 MJCF plant 后才落到物理 pitch/roll/执行器。不要跨变体混用奖励、模型和部署配置。

逐项权重见 [`docs/REWARD_FRAMEWORKS.md`](../../../docs/REWARD_FRAMEWORKS.md)；映射链路见
[`docs/INTERFACE_CONTRACTS.md`](../../../docs/INTERFACE_CONTRACTS.md)。
