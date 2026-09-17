"""Compatibility entry point.

Moved to:
    pitchRoll2UpperLower/model_eval_vis/eval_visualize_pr_ul_models.py
"""

from __future__ import annotations

import runpy
from pathlib import Path
import sys


TARGET = Path(__file__).resolve().parents[1] / "model_eval_vis" / "eval_visualize_pr_ul_models.py"
sys.path.insert(0, str(TARGET.parent))
runpy.run_path(str(TARGET), run_name="__main__")
