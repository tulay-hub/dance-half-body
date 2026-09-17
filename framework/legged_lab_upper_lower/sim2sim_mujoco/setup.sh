#!/bin/bash
# Sim2Sim环境准备脚本
# Isaac Sim → Mujoco 策略迁移

echo "=========================================="
echo "🚀 Sim2Sim环境准备"
echo "=========================================="
echo ""

# 安装mujoco
echo "📦 安装Mujoco..."
pip install mujoco mujoco-python-viewer --quiet

# 验证安装
echo ""
echo "✓ 验证安装..."
python -c "import mujoco; print('  Mujoco版本:', mujoco.__version__)" || exit 1
python -c "import torch; print('  PyTorch版本:', torch.__version__)" || exit 1

echo ""
echo "=========================================="
echo "✅ 环境准备完成"
echo "=========================================="
echo ""
echo "📋 可用命令:"
echo ""
echo "1. 快速测试 (使用最新checkpoint):"
echo "   python run_sim2sim.py"
echo ""
echo "2. 指定checkpoint:"
echo "   python run_sim2sim.py \\"
echo "     --checkpoint ../logs/rsl_rl/lens110_amp/2026-06-26_18-04-38/exported/policy.pt \\"
echo "     --duration 30"
echo ""
echo "3. 使用原始checkpoint对比:"
echo "   python run_sim2sim.py \\"
echo "     --checkpoint ../logs/rsl_rl/<run>/exported/policy.pt"
echo ""
echo "=========================================="
