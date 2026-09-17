#!/bin/bash
# 快速启动脚本 - Sim2Sim测试

cd "$(dirname "$0")"

echo "=========================================="
echo "🚀 Sim2Sim: Isaac Sim → Mujoco"
echo "=========================================="
echo ""

# 检查conda环境
if [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
    echo "使用当前本地 conda 环境: ${CONDA_DEFAULT_ENV}"
else
    echo "未激活 conda 环境，继续使用当前本地 Python 环境"
fi

# 检查依赖
echo "📦 检查依赖..."
python -c "import torch, mujoco" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ 缺少依赖，运行安装..."
    bash setup.sh
fi

echo ""
echo "✅ 环境就绪"
echo ""
echo "=========================================="
echo "📋 可用配置:"
echo "=========================================="
echo "  1. conservative  - 最稳定 (推荐)"
echo "  2. moderate      - 平衡"
echo "  3. aggressive    - 快速 (不推荐)"
echo ""

# 默认使用conservative配置
CONFIG=${1:-conservative}

echo "使用配置: $CONFIG"
echo ""
echo "=========================================="
echo "🎮 启动测试..."
echo "=========================================="
echo ""

# 运行sim2sim
python run_sim2sim.py \
  --checkpoint ../logs/rsl_rl/lens110_amp/2026-06-26_18-04-38/exported/policy.pt \
  --model ../source/legged_lab/legged_lab/data/Robots/model_humanoid_lens110/mjcf/lens110_21dof.xml \
  --config "$CONFIG" \
  --duration 30 \
  --debug

echo ""
echo "=========================================="
echo "✅ 测试完成"
echo "=========================================="
echo ""
echo "💡 提示:"
echo "  - 测试其他配置: bash quick_start.sh moderate"
echo "  - 指定其他本地模型: python run_sim2sim.py --checkpoint ../logs/rsl_rl/<run>/exported/policy.pt"
echo "  - 查看帮助: python run_sim2sim.py --help"
echo ""
