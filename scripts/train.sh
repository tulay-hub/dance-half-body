#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
FRAMEWORK_ROOT="$PROJECT_ROOT/framework/legged_lab_upper_lower"

cd "$FRAMEWORK_ROOT"
exec "${PYTHON:-python}" scripts/rsl_rl/train.py \
  --task LeggedLab-Isaac-AMP-Lens110-UpperLower-v0 "$@"
