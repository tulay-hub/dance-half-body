<p align="center"><a href="#zh">中文</a> &nbsp;|&nbsp; <a href="#en">English</a></p>
<a id="zh"></a>

# 02 · 跳舞半身（Dance Half Body / Upper-Lower）

## 项目定位

这是双足人形机器人的 upper/lower 执行器接口项目。它不是把整个机器人简单砍成若干关节，而是让 policy-facing
ankle IO 使用四个物理 upper/lower 执行器槽位，同时把训练模型中的 pitch/roll 踝状态和动作转换到真实
执行器语义。

实体训练框架：`framework/legged_lab_upper_lower`。

主要任务：

| 变体 | 训练任务 | 核心转换 |
|---|---|---|
| Polynomial | `LeggedLab-Isaac-AMP-Lens110-UpperLower-v0` | pitch/roll ↔ upper/lower 拟合模型 |
| Weighted MLP | `LeggedLab-Isaac-AMP-Lens110-MlpUpperLower-v0` | weighted MLP 状态/力矩转换 |
| XML tendon | `LeggedLab-Isaac-AMP-Lens110-XmlTendonUpperLower-v0` | MJCF tendon geometry solver |
| Direct MJCF | `LeggedLab-Isaac-AMP-Lens110-MJCF-UpperLower-v0` | 直接在 upper/lower plant 上训练 |

每个任务都有对应的 `-Play-v0`，完整注册位置在
`framework/legged_lab_upper_lower/source/legged_lab/legged_lab/tasks/locomotion/amp/config/lens110/__init__.py`。

## 目录

```text
02_dance_half_body/
├── framework/legged_lab_upper_lower/    # 独立 Isaac Lab + AMP 工程
├── data/
│   ├── raw/bvh -> ../dance_dataset/raw_bvh/original_bvh
│   ├── processed/retargeted_actions/mixed_k1_mj_motion/
│   ├── models/pitchRoll2UpperLower -> 框架内映射模型/脚本
│   └── training/motion_data -> 框架内双足人形机器人 motion data
├── exports/training_exports -> AMP runs
├── experiments/runs -> AMP TensorBoard/checkpoint runs
└── docs/
```

## 训练入口

```bash
./scripts/train.sh \
  --headless --num_envs 2048 --max_iterations 200000
```

使用 MLP、XML tendon 或 MJCF 变体时只替换 `--task`，不要拿一个变体的 policy/部署 YAML 去驱动另一种
ankle plant。输出统一归档到本项目 `exports/`，原始/映射数据归档到本项目 `data/`。

## 奖励和接口

任务层奖励包含速度跟踪、站高、姿态、脚步节奏、滑步、脚距、关节限位、力矩和自碰撞；AMP 层另有
discriminator style reward。upper/lower 转换的状态/动作链路见
[`docs/REWARD_FRAMEWORKS.md`](docs/REWARD_FRAMEWORKS.md) 和
[`docs/INTERFACE_CONTRACTS.md`](docs/INTERFACE_CONTRACTS.md)。

这里的 `pitchRoll2UpperLower` 是执行器映射/辅助训练链路，使用前必须检查映射模型版本和左右脚顺序。

## 训练架构

AMP、PPO、LSGAN discriminator、72/21 观测动作接口和四种 upper/lower plant 的完整实现流程见 [`docs/TRAINING_ARCHITECTURE.md`](docs/TRAINING_ARCHITECTURE.md)。当前视频目录没有单独标注为 upper/lower 的演示视频，因此不把通用舞蹈视频误作本项目结果。

<a id="en"></a>

## English

This repository contains the upper/lower half-body motion project. It uses AMP and keeps the policy and physical ankle interfaces explicit: pitch/roll policy commands are converted to upper/lower motor targets by the local mapping models or the XML tendon solver. The verified compact lower-body TorchScript interface is 45 observations to 12 actions; the full policy contract remains 21 actions.

Install the local Isaac Lab and MuJoCo dependencies, then run `./scripts/train.sh --headless --num_envs 2048`. Use the entry points under `framework/legged_lab_upper_lower/sim2sim_mujoco/` for local replay. Never mix a policy, deployment YAML, ankle plant, or mapping model from a different variant.

Before deployment, check the upper/lower joint order, pitch/roll signs, mapping version, actuator limits, checkpoint interface, and replay evidence. Reward and interface details are in `docs/REWARD_FRAMEWORKS.md` and `docs/INTERFACE_CONTRACTS.md`.
