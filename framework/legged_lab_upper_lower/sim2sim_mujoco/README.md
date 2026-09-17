# 🚀 Sim2Sim: Isaac Sim → Mujoco

将训练好的Lens110策略从Isaac Sim迁移到Mujoco仿真器进行快速验证。

## 📁 目录结构

```
sim2sim_mujoco/
├── README.md                      # 本文件
├── run_sim2sim.py                 # 主程序（改进版）
├── setup.sh                       # 环境准备脚本
├── GUIDE.md                       # 详细使用指南
├── TROUBLESHOOTING.md             # 问题诊断文档
└── configs/
    └── domain_adaptation.yaml     # 域适配配置
```

## 🚀 快速开始

### 一键运行（推荐）

```bash
cd sim2sim_mujoco

# 1. 准备环境
bash setup.sh

# 2. 运行测试（保守配置，最稳定）
python run_sim2sim.py --config conservative
```

### 完整命令

```bash
# 激活环境
# Use the active local Python environment with torch and mujoco

# 测试优化后的模型（91k iterations）
python run_sim2sim.py \
  --checkpoint ../logs/rsl_rl/lens110_amp/2026-06-26_18-04-38/exported/policy.pt \
  --config conservative \
  --duration 30 \
  --debug
```

## 📊 配置说明

### Conservative（保守，推荐）✅
- **稳定性**: ⭐⭐⭐⭐⭐
- **速度**: 慢 (0.2-0.4 m/s)
- **适用**: 初次验证，对称性观察
- **参数**: PD缩放60%, 动作缩放60%, 强滤波

### Moderate（中等）
- **稳定性**: ⭐⭐⭐
- **速度**: 中 (0.5-0.7 m/s)
- **适用**: 调优后使用
- **参数**: PD缩放80%, 动作缩放80%, 中等滤波

### Aggressive（激进）
- **稳定性**: ⭐
- **速度**: 快 (0.8+ m/s)
- **适用**: 不推荐（容易摔倒）
- **参数**: 无缩放，轻滤波

## 🎯 主要功能

### 1. 域适配
- ✅ PD参数自动缩放（适应Mujoco物理引擎）
- ✅ 动作幅度调整
- ✅ 动作低通滤波（平滑控制）
- ✅ 自动摔倒检测

### 2. 对比测试
```bash
# 测试原始模型（41k, 可能不对称）
python run_sim2sim.py \
  --checkpoint ../logs/rsl_rl/lens110_amp/2026-06-26_18-04-38/exported/policy.pt \
  --config conservative

# 测试优化模型（91k, 应该对称）
python run_sim2sim.py \
  --checkpoint ../logs/rsl_rl/lens110_amp/2026-06-26_18-04-38/exported/policy.pt \
  --config conservative
```

**观察**: 左右腿步幅是否一致

### 3. 调试模式
```bash
python run_sim2sim.py --config conservative --debug
```

输出详细信息：
- 每步的状态
- 动作幅度统计
- 摔倒原因分析

## ⚠️ 常见问题

### Q1: 机器人摔倒了？
**A**: 正常的sim2sim gap，使用conservative配置：
```bash
python run_sim2sim.py --config conservative
```

### Q2: ModuleNotFoundError: No module named 'torch'?
**A**: 需要激活conda环境：
```bash
# Use the active local Python environment with torch and mujoco
```

### Q3: 找不到checkpoint？
**A**: 使用相对路径：
```bash
python run_sim2sim.py \
  --checkpoint ../logs/rsl_rl/lens110_amp/2026-06-26_18-04-38/exported/policy.pt
```

## 📈 预期结果

### Conservative配置（推荐）
- ✅ 稳定站立30秒
- ✅ 缓慢前进（0.2-0.4 m/s）
- ✅ **步态对称**（左右腿一致）
- ✅ 无摔倒

### 性能对比
| 模型 | 迭代数 | 奖励 | 对称性 | Mujoco表现 |
|------|--------|------|--------|-----------|
| 原始 | 41k | 44.97 | 一般 | 可能不对称 |
| 优化 | 91k | 48.33 | 优秀 | 应该对称 |

## 📚 详细文档

- **GUIDE.md**: 完整使用指南（技术细节、映射关系）
- **TROUBLESHOOTING.md**: 问题诊断与解决方案

## 🔧 高级用法

### 自定义域适配参数

编辑 `configs/domain_adaptation.yaml`:
```yaml
custom:
  kp_scale: 0.7        # PD刚度缩放
  kd_scale: 0.7        # PD阻尼缩放
  action_scale: 0.7    # 动作幅度缩放
  action_filter: 0.4   # 动作滤波系数
```

然后运行：
```bash
python run_sim2sim.py --config custom
```

### 录制视频

Mujoco viewer支持录制：
- 运行时按 **R** 键开始/停止录制
- 视频保存在当前目录

### 批量测试

```bash
# 测试所有checkpoint
for ckpt in ../logs/.../model_*.pt; do
    echo "Testing $ckpt"
    python run_sim2sim.py --checkpoint $ckpt --config conservative --duration 10
done
```

## 🎓 技术说明

### 观测空间（45维）
- 0-2: 基座角速度
- 3-5: 投影重力
- 6-8: 速度命令
- 9-20: 关节位置（相对）
- 21-32: 关节速度
- 33-44: 上一次动作

### 动作空间（12维）
- 12个腿部关节（hip/knee/ankle）
- Scale: 0.25 × action_scale
- 应用到默认位置的偏移

### Sim2Sim Gap来源
1. **物理引擎**: PhysX vs Mujoco (~10-15%性能下降)
2. **接触模型**: Penalty vs Complementarity
3. **求解器**: PGS vs Newton
4. **执行器**: Implicit PD vs Position control

## 📞 支持

遇到问题？查看：
1. `TROUBLESHOOTING.md` - 常见问题解决
2. `GUIDE.md` - 详细技术文档
3. 运行 `python run_sim2sim.py --help`

---

**快速测试命令**:
```bash
# Use the active local Python environment with torch and mujoco && python run_sim2sim.py --config conservative
```

**项目状态**: ✅ 训练完成，对称性优化成功，准备sim2sim验证

**最后更新**: 2026-07-15
