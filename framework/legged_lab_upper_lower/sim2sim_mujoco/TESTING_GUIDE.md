# 🔬 对称性测试完整指南

## 📋 测试目的

对比训练优化前后的模型，验证"一步大一步小"问题是否解决。

---

## 🚀 快速测试（推荐）

### 一键运行对比测试

```bash
cd projects/02_dance_half_body/framework/legged_lab_upper_lower/sim2sim_mujoco
bash test_symmetry.sh
```

**测试流程**：
1. 自动测试原始模型（41k iterations）- 30秒
2. 暂停，等待你按回车
3. 自动测试优化模型（91k iterations）- 30秒
4. 显示对比总结

**观察重点**：
- ✅ 左右腿步幅是否一致
- ✅ 有无"一步大一步小"现象
- ✅ 步态是否流畅自然
- ✅ 横向漂移是否减少

---

## 🔍 详细测试方法

### 方法1: Mujoco对比测试（快速）

#### 测试原始模型（41k）

```bash
cd sim2sim_mujoco

python run_sim2sim.py \
  --checkpoint ../logs/rsl_rl/lens110_amp/2026-06-26_18-04-38/exported/policy.pt \
  --config conservative \
  --duration 30 \
  --debug
```

**观察**：
- 步态是否不对称
- 是否有"一步大一步小"
- 横向漂移情况

#### 测试优化模型（91k）

```bash
python run_sim2sim.py \
  --checkpoint ../logs/rsl_rl/lens110_amp/2026-06-26_18-04-38/exported/policy.pt \
  --config conservative \
  --duration 30 \
  --debug
```

**对比**：
- 步态是否更对称
- "一步大一步小"是否消失
- 整体是否更流畅

### 方法2: Isaac Sim测试（更清晰）

Isaac Sim的可视化更清晰，更容易观察细节。

#### 测试原始模型

```bash
cd projects/02_dance_half_body/framework/legged_lab_upper_lower

python scripts/rsl_rl/play_single_keyboard.py \
  --task LeggedLab-Isaac-AMP-Lens110-LegPitchRoll-Play-v0 \
  --num_envs 1 \
  --load_run 2026-07-14_13-31-35
```

#### 测试优化模型

```bash
python scripts/rsl_rl/play_single_keyboard.py \
  --task LeggedLab-Isaac-AMP-Lens110-LegPitchRoll-Play-v0 \
  --num_envs 1 \
  --load_run 2026-07-15_17-39-11
```

**Isaac Sim优势**：
- ✅ 高质量3D渲染
- ✅ 可以调整摄像机角度
- ✅ 更容易看清细节
- ✅ 支持慢动作

### 方法3: 录制视频对比（最准确）

#### 在Mujoco中录制

1. 运行测试
2. 按 **R** 键开始录制
3. 等待30秒
4. 再按 **R** 键停止录制
5. 视频保存在当前目录

```bash
# 录制原始模型
python run_sim2sim.py \
  --checkpoint logs/rsl_rl/lens110_amp/2026-06-26_18-04-38/exported/policy.pt \
  --config conservative \
  --duration 30

# 录制优化模型
python run_sim2sim.py \
  --checkpoint logs/rsl_rl/lens110_amp/2026-06-26_18-04-38/exported/policy.pt \
  --config conservative \
  --duration 30
```

#### 视频分析

使用视频播放器：
- 慢放观察步态
- 暂停对比特定帧
- 并排播放两个视频

---

## 📊 对比检查清单

### 1. 步态对称性 ⭐⭐⭐⭐⭐

**检查项目**：
- [ ] 左右腿步幅是否相等
- [ ] 左右腿腾空时间是否一致
- [ ] 左右腿着地力度是否相同

**预期结果**：
- 原始模型：可能不对称
- 优化模型：应该对称

### 2. 步幅一致性

**检查项目**：
- [ ] 每步距离是否稳定
- [ ] 是否有"一步大一步小"
- [ ] 步频是否均匀

**预期结果**：
- 原始模型：可能有大小步
- 优化模型：步幅应该一致

### 3. 横向稳定性

**检查项目**：
- [ ] Y方向漂移是否小
- [ ] 是否有左右摇摆
- [ ] 行走轨迹是否直线

**预期结果**：
- 优化模型应该更稳定

### 4. 整体流畅度

**检查项目**：
- [ ] 动作是否平滑
- [ ] 是否有抖动
- [ ] 是否有突然的动作

**预期结果**：
- 优化模型应该更流畅

---

## 📈 定量分析

### 查看训练指标对比

```bash
# 原始模型指标（从之前的分析）
原始 (41k):
  - 平均奖励: 44.97
  - hip_yaw_symmetry: ~-0.08 (配置权重)
  
# 优化模型指标
优化 (91k):
  - 平均奖励: 48.33 (+7.5%)
  - hip_yaw_symmetry: -0.004 (实际惩罚，极佳)
  - feet_air_time: 0.088 (规律步态)
  - feet_slide: -0.056 (滑步少)
```

**关键指标**：
- `hip_yaw_symmetry`: -0.004 vs ~-0.08
  - 降低95% = 对称性提升95%！

### 使用TensorBoard对比

```bash
tensorboard --logdir logs/rsl_rl/lens110_amp_leg_pitch_roll/
```

在浏览器中对比：
- `Episode_Reward/hip_yaw_symmetry` 曲线
- `Episode_Reward/feet_air_time` 曲线
- `Train/mean_reward` 曲线

---

## 🎯 测试场景

### 场景1: 直线前进（基础）

```bash
# Conservative配置，速度0.3-0.5 m/s
bash test_symmetry.sh
```

**观察**：基础步态对称性

### 场景2: 较快速度（进阶）

```bash
# Moderate配置，速度0.5-0.7 m/s
python run_sim2sim.py --config moderate
```

**观察**：高速下的对称性保持

### 场景3: 多环境并行（全面）

```bash
cd projects/02_dance_half_body/framework/legged_lab_upper_lower

python scripts/rsl_rl/play.py \
  --task LeggedLab-Isaac-AMP-Lens110-LegPitchRoll-Play-v0 \
  --num_envs 16 \
  --load_run 2026-07-15_17-39-11
```

**观察**：16个机器人同时运行，统计对称性

---

## 💡 测试技巧

### 1. 侧视角观察

Mujoco viewer中：
- 鼠标右键拖拽调整视角
- 从侧面观察最容易看出步幅差异

### 2. 慢动作播放

- 录制视频后用播放器慢放
- 0.25x或0.5x速度最佳

### 3. 标记参考点

- 注意观察地面格子
- 数每步跨过几个格子
- 对比左右腿

### 4. 多次测试

- 每个模型至少测试3次
- 排除随机性影响
- 记录平均表现

---

## 📝 测试报告模板

```markdown
# 对称性测试报告

## 测试环境
- 日期: 2026-07-16
- 平台: Mujoco
- 配置: conservative

## 历史模型 A（41k，当前整理目录未保留 checkpoint）
- Checkpoint: `model_41100.pt`（仅历史记录，不是当前本地可用文件）
- 平均奖励: 44.97

### 观察结果
- 步态对称性: [ ]/5
- 步幅一致性: [ ]/5
- 横向稳定性: [ ]/5
- 整体流畅度: [ ]/5

### 问题
- [ ] 一步大一步小
- [ ] 横向漂移
- [ ] 步态不规律
- [ ] 其他: ___________

## 历史模型 B（91k，当前整理目录未保留 checkpoint）
- Checkpoint: `model_91000.pt`（仅历史记录，不是当前本地可用文件）
- 平均奖励: 48.33 (+7.5%)

### 观察结果
- 步态对称性: [ ]/5
- 步幅一致性: [ ]/5
- 横向稳定性: [ ]/5
- 整体流畅度: [ ]/5

### 改善
- [ ] 对称性显著提升
- [ ] 步幅更一致
- [ ] 横向更稳定
- [ ] 整体更流畅

## 结论
- 对称性优化是否成功: [是/否]
- 主要改善: ___________
- 建议: ___________
```

---

## 🚀 快速命令参考

```bash
# 1. 一键对比测试
cd sim2sim_mujoco && bash test_symmetry.sh

# 2. 单独测试原始模型
python run_sim2sim.py \
  --checkpoint logs/rsl_rl/lens110_amp/2026-06-26_18-04-38/exported/policy.pt \
  --config conservative

# 3. 单独测试优化模型
python run_sim2sim.py \
  --checkpoint logs/rsl_rl/lens110_amp/2026-06-26_18-04-38/exported/policy.pt \
  --config conservative

# 4. Isaac Sim测试（更清晰）
cd .. && python scripts/rsl_rl/play_single_keyboard.py \
  --task LeggedLab-Isaac-AMP-Lens110-LegPitchRoll-Play-v0 \
  --num_envs 1 \
  --load_run 2026-07-15_17-39-11

# 5. 尝试更快速度
cd sim2sim_mujoco && bash quick_start.sh moderate
```

---

**现在开始测试吧！**

```bash
cd sim2sim_mujoco
bash test_symmetry.sh
```

观察对比两个模型的步态，看对称性是否真的改善了！🚀
