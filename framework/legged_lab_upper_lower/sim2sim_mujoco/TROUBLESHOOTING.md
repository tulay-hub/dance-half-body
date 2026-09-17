# 🔍 Sim2Sim问题诊断与解决方案

## 问题1: 机器人在Mujoco中摔倒

### 观察到的现象
```
Step   100 | Pos: [-0.002,  0.000,  0.658] | Vel: [-0.025, -0.000]  ✓ 正常
Step   700 | Pos: [-0.139, -0.000,  0.647] | Vel: [-0.311,  0.000]  ✓ 正常
Step   800 | Pos: [-0.230, -0.000,  0.629] | Vel: [-0.631, -0.000]  ⚠ 速度激增
Step   900 | Pos: [-0.409, -0.000,  0.552] | Vel: [-1.186, -0.000]  ❌ 失控
Step  1000 | Pos: [-0.679, -0.000,  0.255] | Vel: [-1.322, -0.000]  ❌ 摔倒
```

### 根本原因

**Sim2Sim Gap**：Isaac Sim和Mujoco的物理引擎差异

| 差异项 | Isaac Sim (PhysX) | Mujoco | 影响 |
|--------|-------------------|---------|------|
| **接触模型** | Penalty-based | Complementarity | 高 |
| **求解器** | PGS | Newton | 高 |
| **积分器** | Semi-implicit Euler | Runge-Kutta | 中 |
| **摩擦模型** | Pyramid friction cone | Box friction cone | 中 |
| **执行器模型** | Implicit PD | Position control | 高 |

## 解决方案

### 方案1: 保守配置（推荐）✅

使用改进版脚本的conservative模式：

```bash
# 使用包含 torch 和 mujoco 的本地 Python 环境运行
# Use the active local Python environment with torch and mujoco

cd projects/02_dance_half_body/framework/legged_lab_upper_lower/sim2sim_mujoco
python run_sim2sim.py \
  --checkpoint ../logs/rsl_rl/lens110_amp/2026-06-26_18-04-38/exported/policy.pt \
  --config conservative \
  --duration 30 \
  --debug
```

**保守配置参数**:
- `kp_scale`: 0.6 (PD刚度降低40%)
- `kd_scale`: 0.6 (PD阻尼降低40%)
- `action_scale`: 0.6 (动作幅度降低40%)
- `action_filter`: 0.5 (强动作滤波)

**预期效果**: 更稳定但行走可能较慢

### 方案2: 中等配置

```bash
python run_sim2sim.py \
  --checkpoint ../logs/rsl_rl/lens110_amp/2026-06-26_18-04-38/exported/policy.pt \
  --config moderate \
  --duration 30
```

**中等配置参数**:
- `kp_scale`: 0.8
- `kd_scale`: 0.8
- `action_scale`: 0.8
- `action_filter`: 0.3

### 方案3: 训练时增加Domain Randomization

**长期解决方案**: 重新训练时增加物理参数随机化

修改配置文件增加：
```python
# 在训练配置中
self.events.randomize_physics = EventCfg(
    func=mdp.randomize_physics,
    params={
        'friction_range': (0.3, 1.8),      # 增加摩擦力范围
        'restitution_range': (0.0, 0.3),   # 增加恢复系数范围
        'contact_stiffness_range': (0.5, 2.0),  # 接触刚度
    }
)
```

### 方案4: Mujoco参数调整

手动调整Mujoco模型参数使其接近Isaac Sim：

```xml
<!-- 在lens110_21dof.xml中 -->
<option>
  <flag contact="enable"/>
  <flag gravity="enable"/>
  <!-- 调整求解器参数 -->
  <solver iterations="100"/>  <!-- 增加迭代次数 -->
  <integrator>RK4</integrator>  <!-- 使用RK4积分器 -->
</option>

<!-- 调整执行器参数 -->
<actuator>
  <position joint="left_hip_pitch" kp="32" kd="4"/>  <!-- 80%的Isaac值 -->
  <!-- 其他关节类似... -->
</actuator>

<!-- 调整接触参数 -->
<geom ... friction="1.0 0.05 0.005"/>  <!-- 匹配Isaac摩擦力 -->
```

## 问题2: ModuleNotFoundError: No module named 'torch'

### 原因
在系统Python而非conda环境中运行

### 解决方案

```bash
# 1. 激活正确的conda环境
# Use the active local Python environment with torch and mujoco

# 2. 验证torch可用
python -c "import torch; print('PyTorch:', torch.__version__)"

# 3. 运行sim2sim
python run_sim2sim.py --config conservative
```

## 完整测试流程

### 第一步: 环境检查

```bash
# 激活环境
# Use the active local Python environment with torch and mujoco

# 检查依赖
python -c "import torch, mujoco; print('✓ 依赖OK')"

# 如果mujoco未安装
pip install mujoco mujoco-python-viewer
```

### 第二步: 保守测试

```bash
cd projects/02_dance_half_body/framework/legged_lab_upper_lower/sim2sim_mujoco

# 测试优化后模型（conservative配置）
python run_sim2sim.py \
  --checkpoint ../logs/rsl_rl/lens110_amp/2026-06-26_18-04-38/exported/policy.pt \
  --config conservative \
  --duration 30 \
  --debug
```

### 第三步: 对比测试

如果conservative配置稳定，尝试moderate配置：

```bash
python run_sim2sim.py \
  --checkpoint ../logs/rsl_rl/lens110_amp/2026-06-26_18-04-38/exported/policy.pt \
  --config moderate \
  --duration 30
```

### 第四步: 原始模型对比

测试对称性优化前的模型：

```bash
python run_sim2sim.py \
  --checkpoint ../logs/rsl_rl/lens110_amp/2026-06-26_18-04-38/exported/policy.pt \
  --config conservative \
  --duration 30
```

## 预期结果

### Conservative配置
- ✅ 应该能稳定站立30秒
- ✅ 可能行走较慢（0.2-0.4 m/s）
- ✅ 步态对称性应该可见

### Moderate配置
- ⚠ 可能在15-20秒后不稳定
- ⚠ 行走速度接近训练目标（0.5-0.8 m/s）
- ⚠ 需要调试PD参数

### Aggressive配置
- ❌ 很可能快速摔倒（类似原始测试）
- ❌ 不推荐使用

## 性能对比表

| 配置 | 稳定性 | 行走速度 | 对称性可见度 | 推荐场景 |
|------|--------|----------|-------------|----------|
| **Conservative** | ⭐⭐⭐⭐⭐ | 慢 (0.2-0.4) | ⭐⭐⭐⭐ | 初次验证 |
| **Moderate** | ⭐⭐⭐ | 中 (0.5-0.7) | ⭐⭐⭐⭐⭐ | 调优后 |
| **Aggressive** | ⭐ | 快 (0.8+) | ⭐ | 不推荐 |

## 调试技巧

### 1. 观察关键指标

```python
# 在sim2sim脚本中添加
if step_count % 50 == 0:
    print(f"Height: {base_pos[2]:.3f}, "
          f"Tilt: {tilt_angle*180/np.pi:.1f}°, "
          f"Action: {np.abs(action).max():.3f}")
```

### 2. 录制视频

```bash
# Mujoco viewer支持录制
# 运行时按 'R' 键开始/停止录制
```

### 3. 逐步增加难度

```bash
# 第1次: conservative, 速度0.3
# 第2次: conservative, 速度0.5
# 第3次: moderate, 速度0.5
# 第4次: moderate, 速度0.8
```

## 下一步建议

### 如果Conservative配置成功 ✅

1. **记录成功参数**: 保存成功的config设置
2. **测试对称性**: 观察左右腿步幅是否一致
3. **逐步提速**: 从0.3逐步增加到0.8 m/s
4. **导出ONNX**: 准备真机部署

### 如果仍然失败 ❌

1. **检查Mujoco模型**: 确认MJCF文件正确
2. **降低学习率**: 使用更早的checkpoint（如model_50000.pt）
3. **手动调参**: 直接修改DOMAIN_ADAPTATION参数
4. **重新训练**: 增加Domain Randomization后重训

## 参考命令总结

```bash
# 环境准备
# Use the active local Python environment with torch and mujoco
pip install mujoco mujoco-python-viewer

# 保守测试（推荐）
python run_sim2sim.py --config conservative --debug

# 中等测试
python run_sim2sim.py --config moderate

# 原始模型对比
python run_sim2sim.py \
  --checkpoint ../logs/rsl_rl/lens110_amp/2026-06-26_18-04-38/exported/policy.pt \
  --config conservative
```

---

**关键结论**: 
- 当前摔倒是**正常的sim2sim gap**，不是训练问题
- 使用**conservative配置**应该能解决
- 对称性优化的效果需要在稳定行走的基础上观察

**最后更新**: 2026-07-15
