# PR/UL Model Evaluation And Visualization

## Conversion UI

Qt conversion UI:

```bash
cd projects/02_dance_half_body/framework/legged_lab_upper_lower
python -B pitchRoll2UpperLower/model_eval_vis/ankle_mlp_converter_qt.py
```

Web conversion UI:

```bash
cd projects/02_dance_half_body/framework/legged_lab_upper_lower
python -B pitchRoll2UpperLower/model_eval_vis/ankle_mlp_converter_ui.py --port 7862
```

MuJoCo initial pose viewer:

```bash
cd projects/02_dance_half_body/framework/legged_lab_upper_lower
python -B pitchRoll2UpperLower/model_eval_vis/view_lens110_initial_poses.py
```

The old `pitchRoll2UpperLower/t_p_v/...` paths are kept as compatibility entry points.

## CSV Evaluation Browser

This opens a Qt window to browse `summary.csv`, `report.html`, scatter plots,
error histograms, and time-series plots generated from the test CSV data.

```bash
cd projects/02_dance_half_body/framework/legged_lab_upper_lower
python -B pitchRoll2UpperLower/model_eval_vis/model_compare_qt.py
```

If the active conda env has no Qt/matplotlib/torch, the script relaunches with:

```bash
python
```

The default eval result folder is:

```text
pitchRoll2UpperLower/model_eval_vis/csv_eval_vis
```

Click `Run CSV eval` in the UI if the result folder is missing or stale.

The same window also has an `Ankle model` page for the legacy data under:

```text
pitchRoll2UpperLower/ankle_model
```

Its generated result folder is:

```text
pitchRoll2UpperLower/model_eval_vis/ankle_model_eval_vis
```

Click `Run ankle_model eval` on that page to regenerate it.

## CSV Batch Evaluation

This keeps the previous dataset-based metrics and plots, but writes outputs under this folder.

```bash
cd projects/02_dance_half_body/framework/legged_lab_upper_lower
env -u PYTHONPATH -u PYTHONHOME python -B \
  pitchRoll2UpperLower/model_eval_vis/eval_visualize_pr_ul_models.py
```

Default output:

```text
pitchRoll2UpperLower/model_eval_vis/csv_eval_vis
```
