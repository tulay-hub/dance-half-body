# 🚀 Sim2Sim: Isaac Sim → Mujoco 策略迁移指南

## 概述

将在Isaac Sim中训练的Lens110人形机器人策略迁移到Mujoco仿真器进行快速验证。

## 📋 准备工作

### 1. 安装依赖

```bash
cd projects/02_dance_half_body/framework/legged_lab_upper_lower/sim2sim_mujoco
bash setup.sh
```

或手动安装：

```bash
pip install mujoco mujoco-python-viewer
```

### 2. 验证文件

确保以下文件存在：
- ✓ Mujoco模型: `source/legged_lab/legged_lab/data/Robots/model_humanoid_lens110/mjcf/lens110_21dof.xml`
- ✓ 训练checkpoint: `../logs/rsl_rl/lens110_amp/2026-06-26_18-04-38/exported/policy.pt`

## 🎮 使用方法

### 快速开始 (推荐)

测试对称性优化后的模型（91k iterations）：

```bash
cd projects/02_dance_half_body/framework/legged_lab_upper_lower/sim2sim_mujoco

python run_sim2sim.py
```

### 对比测试

**测试原始模型（41k iterations）：**

```bash
python run_sim2sim.py \
  --checkpoint ../logs/rsl_rl/lens110_amp/2026-06-26_18-04-38/exported/policy.pt \
  --duration 30
```

**测试优化后模型（91k iterations）：**

```bash
python run_sim2sim.py \
  --checkpoint ../logs/rsl_rl/lens110_amp/2026-06-26_18-04-38/exported/policy.pt \
  --duration 30
```

### 高级参数

```bash
python run_sim2sim.py \
  --checkpoint <checkpoint路径> \
  --model <mujoco模型路径> \
  --duration <运行秒数>
```

## 📊 关键观察指标

运行sim2sim时，注意观察：

### 1. 步态对称性
- ✓ 左右腿步幅是否一致（对称性优化的核心）
- ✓ 步态节奏是否流畅
- ✓ 没有"一步大一步小"现象

### 2. 稳定性
- ✓ 机器人是否能保持直立行走
- ✓ 基座高度是否稳定在0.68m左右
- ✓ 是否有频繁摔倒

### 3. 运动质量
- ✓ 足部是否有滑动
- ✓ 手臂摆动是否自然
- ✓ 整体动作是否流畅

## 🔧 技术细节

### 观测空间映射

从Mujoco提取的45维观测：

| 维度 | 内容 | 说明 |
|------|------|------|
| 0-2 | base_ang_vel | 基座角速度 |
| 3-5 | projected_gravity | 投影重力 |
| 6-8 | velocity_commands | 速度命令 [vx, vy, vyaw] |
| 9-20 | joint_pos (relative) | 12个腿部关节位置（相对默认值） |
| 21-32 | joint_vel | 12个腿部关节速度 |
| 33-44 | last_action | 上一次动作 |

### 动作空间映射

策略输出12维动作（腿部关节）：
- Scale: 0.25
- 添加到默认关节位置
- 通过PD控制器执行

### 主要差异与适配

| 项目 | Isaac Sim | Mujoco | 适配方案 |
|------|-----------|--------|----------|
| **物理引擎** | PhysX | Mujoco | 参数调整 |
| **控制频率** | 50Hz | 可配置 | 统一到50Hz |
| **关节名称** | Isaac格式 | MJCF格式 | 映射表转换 |
| **坐标系** | USD | Mujoco | 旋转矩阵转换 |
| **执行器** | Implicit PD | 位置控制 | PD参数匹配 |

## ⚠️ 已知限制

1. **手臂控制**: 当前脚本未实现脚本化手臂摆动，手臂保持默认姿态
2. **地形**: 仅支持平地，不支持复杂地形
3. **扰动**: Mujoco中未添加推力扰动
4. **传感器噪声**: 观测无噪声（可能导致sim2sim gap）

## 🎯 性能对比

### 预期结果

**原始模型（41k, 奖励44.97）**:
- 可能出现不对称步态
- 整体能行走但质量一般

**优化后模型（91k, 奖励48.33）**:
- ✓ 步态对称性显著改善
- ✓ 更流畅的运动
- ✓ 更少的滑步

### Sim2Sim Gap

从Isaac Sim到Mujoco可能存在的性能下降：
- 物理引擎差异: ~5-10%
- 执行器模型差异: ~3-5%
- 传感器噪声缺失: ~2-3%

**总体预期**: Mujoco中的表现可能比Isaac Sim低10-15%，但整体行为应该一致。

## 📈 下一步

### 如果Mujoco中表现良好：

1. **导出ONNX模型**（便于部署）:
   ```python
   # 在sim2sim脚本中添加导出功能
   torch.onnx.export(policy, dummy_input, "lens110_policy.onnx")
   ```

2. **Sim2Real准备**:
   - 增加观测噪声
   - 增加动作延迟
   - 测试不同地面摩擦力
   - 添加系统辨识误差

3. **真机部署**:
   - 将ONNX模型部署到机器人控制器
   - 实施安全机制（姿态保护、紧急停止）
   - 逐步增加速度指令

### 如果性能不理想：

1. **调试观测映射**: 检查Mujoco观测是否正确
2. **参数对齐**: 确保PD参数与Isaac Sim一致
3. **Domain Randomization**: 在训练时增加更多随机化
4. **Fine-tune**: 在Mujoco中继续训练几千步

## 💡 常见问题

**Q: 机器人立即倒下？**
A: 检查初始高度设置，确保为0.68m

**Q: 动作很奇怪/抖动？**
A: 可能是关节映射错误或PD参数不匹配

**Q: 性能远低于Isaac Sim？**
A: 正常的sim2sim gap，可以通过增加训练时的随机化改善

**Q: 如何录制视频？**
A: Mujoco viewer支持录制，或使用screen recording工具

## 📚 参考资料

- [Mujoco文档](https://mujoco.readthedocs.io/)
- [Isaac Lab文档](https://isaac-sim.github.io/IsaacLab/)
- [Sim2Real论文集](https://github.com/topics/sim-to-real)

---

**最后更新**: 2026-07-15
**状态**: 已完成对称性优化，准备sim2sim验证
