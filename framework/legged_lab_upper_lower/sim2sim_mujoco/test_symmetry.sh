#!/bin/bash
# 对称性对比测试脚本
# 对比两个由用户显式提供的本地模型在 MuJoCo 中的步态表现。
# 若只给一个参数，则两次使用同一当前本地 21DoF 策略，仅做 replay smoke。

cd "$(dirname "$0")"

ORIGINAL_CHECKPOINT="${1:-../logs/rsl_rl/lens110_amp/2026-06-26_18-04-38/exported/policy.pt}"
OPTIMIZED_CHECKPOINT="${2:-${ORIGINAL_CHECKPOINT}}"

echo "=========================================="
echo "🔬 对称性对比测试"
echo "=========================================="
echo ""
echo "将测试以下模型："
echo "  1. 模型 A: ${ORIGINAL_CHECKPOINT}"
echo "  2. 模型 B: ${OPTIMIZED_CHECKPOINT}"
echo ""
echo "测试时长: 各30秒"
echo "配置: conservative (最稳定)"
echo ""
echo "=========================================="
echo ""

# 检查环境
if [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
    echo "使用当前本地 conda 环境: ${CONDA_DEFAULT_ENV}"
else
    echo "未激活 conda 环境，继续使用当前本地 Python 环境"
fi

# 测试1: 模型 A
echo "=========================================="
echo "📊 测试 1/2: 模型 A"
echo "=========================================="
echo ""
echo "⏳ 运行30秒，请观察步态是否对称..."
echo ""

python run_sim2sim.py \
  --checkpoint "$ORIGINAL_CHECKPOINT" \
  --config conservative \
  --duration 30 \
  --debug

echo ""
echo "✓ 原始模型测试完成"
echo ""
echo "请回答以下问题："
echo "  - 左右腿步幅是否一致？ (Y/n)"
echo "  - 有无'一步大一步小'现象？ (Y/n)"
echo ""
read -p "按回车继续测试优化模型..."

echo ""
echo ""

# 测试2: 模型 B
echo "=========================================="
echo "📊 测试 2/2: 模型 B"
echo "=========================================="
echo ""
echo "⏳ 运行30秒，请观察步态改善..."
echo ""

python run_sim2sim.py \
  --checkpoint "$OPTIMIZED_CHECKPOINT" \
  --config conservative \
  --duration 30 \
  --debug

echo ""
echo "✓ 优化模型测试完成"
echo ""

# 总结
echo "=========================================="
echo "📊 测试总结"
echo "=========================================="
echo ""
echo "请对比两次测试的观察结果："
echo ""
echo "1. 步态对称性："
echo "   - 原始模型: [  ]"
echo "   - 优化模型: [  ]"
echo ""
echo "2. 步幅一致性："
echo "   - 原始模型: [  ]"
echo "   - 优化模型: [  ]"
echo ""
echo "3. 整体流畅度："
echo "   - 原始模型: [  ]"
echo "   - 优化模型: [  ]"
echo ""
echo "4. 稳定性："
echo "   - 原始模型: [  ]"
echo "   - 优化模型: [  ]"
echo ""
echo "=========================================="
echo ""
echo "💡 提示："
echo "  - 如果差异不明显，可以录制视频后慢放对比"
echo "  - 按 R 键在Mujoco viewer中录制视频"
echo "  - 也可以在Isaac Sim中测试观察更清晰"
echo ""
echo "下一步建议："
echo "  1. 如果对称性改善明显 → 尝试moderate配置提速"
echo "  2. 如果差异不大 → 在Isaac Sim中测试"
echo "  3. 准备真机部署 → 导出ONNX模型"
echo ""
