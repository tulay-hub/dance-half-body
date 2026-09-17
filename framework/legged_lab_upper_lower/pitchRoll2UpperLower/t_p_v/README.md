# Lens110 T/P/V trajectory collection

Collect scripted MuJoCo trajectories for the Lens110 upper/lower ankle motors.
The script directly excites `left/right_ankle_upper_joint` and
`left/right_ankle_lower_joint`, then records the passive ankle
`pitch/roll` response together with motor torque, position, and velocity.

Default output:

```bash
python -u pitchRoll2UpperLower/t_p_v/collect_t_p_v_trajectories.py --overwrite
```

The default run writes:

```text
pitchRoll2UpperLower/t_p_v/data/lens110_t_p_v_trajectories.csv
```

Scenarios:

- `stand_motion`: standing arm swing or in-place march, non-ankle joints move.
- `left_support_right_high_knee_ankle`: left support, right leg high-knee, right ankle moves.
- `right_support_left_high_knee_ankle`: right support, left leg high-knee, left ankle moves.
- `stand_ankle_only`: standing pose, non-ankle joints fixed, both ankles move.

Useful examples:

Watch MuJoCo while collecting data:

```bash
python -u pitchRoll2UpperLower/t_p_v/collect_t_p_v_trajectories.py \
  --scenario stand_ankle_only \
  --render_mode window \
  --realtime \
  --camera_azimuth 90 \
  --output_csv pitchRoll2UpperLower/t_p_v/data/watch_stand_ankle_only.csv \
  --overwrite
```

Record preview video while collecting data:

```bash
python -u pitchRoll2UpperLower/t_p_v/collect_t_p_v_trajectories.py \
  --scenario left_support_right_high_knee_ankle \
  --render_mode video \
  --camera_azimuth 90 \
  --video_path pitchRoll2UpperLower/t_p_v/data/high_knee_preview.mp4 \
  --output_csv pitchRoll2UpperLower/t_p_v/data/high_knee_tpv.csv \
  --overwrite
```

If the ankle motion is still too small visually, increase `--ankle_upper_amp`,
`--ankle_lower_amp`, or `--ankle_motor_kp_scale`.  The script directly commands
the upper/lower motor targets; pitch/roll joints are recorded as the linkage
response, not used as the command.

```bash
python -u pitchRoll2UpperLower/t_p_v/collect_t_p_v_trajectories.py \
  --scenario stand_ankle_only \
  --sim_duration 30 \
  --output_csv pitchRoll2UpperLower/t_p_v/data/stand_ankle_only.csv \
  --overwrite
```

```bash
python -u pitchRoll2UpperLower/t_p_v/collect_t_p_v_trajectories.py \
  --scenario stand_motion \
  --stand_motion_style arm_swing \
  --output_csv pitchRoll2UpperLower/t_p_v/data/stand_arm_swing.csv \
  --overwrite
```

CSV fields include:

- `q_*`, `dq_*`, `tau_*`: actual actuated control-joint position, velocity, and torque.
- `target_q_*`, `target_dq_*`: commanded control-joint target position and finite-difference target velocity.
- `obs_q_*`, `obs_dq_*`: observed pitch/roll joint positions and velocities, including passive ankle pitch/roll.
- `obs_target_q_*`, `obs_target_dq_*`: body-pose targets for non-ankle joints.  The ankle pitch/roll target columns stay at the held pose while upper/lower motors are directly excited.

For training the upper/lower to pitch/roll relation, use the four upper/lower
motor columns as inputs and the passive pitch/roll columns as outputs, for
example:

- Inputs: `q_left_ankle_upper_joint`, `q_left_ankle_lower_joint`, `dq_left_ankle_upper_joint`, `dq_left_ankle_lower_joint`, `tau_left_ankle_upper_joint`, `tau_left_ankle_lower_joint`
- Outputs: `obs_q_left_ankle_pitch_joint`, `obs_q_left_ankle_roll_joint`, `obs_dq_left_ankle_pitch_joint`, `obs_dq_left_ankle_roll_joint`

## Policy-driven pitch/roll <-> upper/lower collection

For free-base data, prefer the trained pitch/roll policy instead of replaying
retargeted pkl files directly.  The policy keeps the robot balanced in MuJoCo;
the script records the simulated pitch/roll ankle position, velocity, and
torque, then maps the same state through the real Lens110 upper/lower tendon
geometry.  Upper/lower torque columns are virtual-work equivalent torques, not
directly simulated upper/lower motor readings.

Watch MuJoCo while collecting:

```bash
cd projects/02_dance_half_body/framework/legged_lab_upper_lower

TMPDIR=/tmp PYTHONDONTWRITEBYTECODE=1 \
python -u \
  pitchRoll2UpperLower/t_p_v/collect_policy_pr_ul_data.py \
  --load_onnx logs/rsl_rl/lens110_amp/2026-06-26_18-04-38/exported/policy.onnx \
  --policy_backend onnx \
  --cmd_vel 0.4 0.0 0.0 \
  --sim_duration 60 \
  --render_mode window \
  --realtime \
  --output_csv pitchRoll2UpperLower/t_p_v/data/policy_pr_ul_walk.csv \
  --overwrite
```

Headless collection:

```bash
TMPDIR=/tmp PYTHONDONTWRITEBYTECODE=1 \
python -u \
  pitchRoll2UpperLower/t_p_v/collect_policy_pr_ul_data.py \
  --load_onnx logs/rsl_rl/lens110_amp/2026-06-26_18-04-38/exported/policy.onnx \
  --policy_backend onnx \
  --cmd_vel 0.4 0.0 0.0 \
  --sim_duration 120 \
  --render_mode none \
  --output_csv pitchRoll2UpperLower/t_p_v/data/policy_pr_ul_walk.csv \
  --overwrite
```

Useful runtime keys in window mode:

```text
8/2: vx +/-
4/6: vy +/-
7/9: yaw rate +/-
arrow up/down/left/right: vx/yaw shortcuts
0: reset
F: camera follow
```

The CSV is compatible with the relation training scripts and includes:

```text
left_pitch_pos, left_pitch_vel, left_pitch_tau
left_roll_pos, left_roll_vel, left_roll_tau
right_pitch_pos, right_pitch_vel, right_pitch_tau
right_roll_pos, right_roll_vel, right_roll_tau

left_upper_pos, left_upper_vel, left_upper_tau
left_lower_pos, left_lower_vel, left_lower_tau
right_upper_pos, right_upper_vel, right_upper_tau
right_lower_pos, right_lower_vel, right_lower_tau
```

## Replay retargeted pkl trajectories

This path is mainly for inspection/debugging.  Retargeted kinematic clips are
not dynamically stabilized policies, so direct free-base PD replay can fall
over quickly and can produce poor training data.  The retargeted Lens110
stand-leg trajectories are located at:

```text
source/legged_lab/legged_lab/data/MotionData/lens110_gmr_lab
```

This records ankle pitch/roll and upper/lower motor position, velocity, and
torque into one CSV:

Use `--pin_root` only for debugging a fixed replay.  Add
`--use_motion_root_rotation` only when you want to inspect the original root
orientation.

```bash
cd projects/02_dance_half_body/framework/legged_lab_upper_lower

TMPDIR=/tmp PYTHONDONTWRITEBYTECODE=1 \
python -u \
  pitchRoll2UpperLower/t_p_v/replay_retargeted_pr_ul_data.py \
  --motion source/legged_lab/legged_lab/data/MotionData/lens110_gmr_lab \
  --output_csv pitchRoll2UpperLower/t_p_v/data/retarget_pkl_pr_ul.csv \
  --overwrite
```

Preview one motion:

```bash
TMPDIR=/tmp PYTHONDONTWRITEBYTECODE=1 \
python -u \
  pitchRoll2UpperLower/t_p_v/replay_retargeted_pr_ul_data.py \
  --motion source/legged_lab/legged_lab/data/MotionData/lens110_gmr_lab/02_02_stageii_walk.pkl \
  --render_mode window \
  --realtime \
  --output_csv pitchRoll2UpperLower/t_p_v/data/watch_49_18_pr_ul.csv \
  --overwrite
```

Useful output columns:

```text
left_pitch_pos, left_pitch_vel, left_pitch_tau
left_roll_pos, left_roll_vel, left_roll_tau
right_pitch_pos, right_pitch_vel, right_pitch_tau
right_roll_pos, right_roll_vel, right_roll_tau

left_upper_pos, left_upper_vel, left_upper_tau
left_lower_pos, left_lower_vel, left_lower_tau
right_upper_pos, right_upper_vel, right_upper_tau
right_lower_pos, right_lower_vel, right_lower_tau
```

## Train pitch/roll <-> upper/lower relation models

Polynomial ridge model:

```bash
TMPDIR=/tmp CONDA_NO_PLUGINS=true PYTHONDONTWRITEBYTECODE=1 \
python -u \
  pitchRoll2UpperLower/t_p_v/train_ankle_pr_ul_relation_poly.py \
  --csv pitchRoll2UpperLower/t_p_v/data/policy_pr_ul_walk.csv \
  --relation pr_to_ul_full \
  --save-dir pitchRoll2UpperLower/t_p_v/models_pr_ul_relation_poly \
  --degree 3 \
  --overwrite
```

MLP model:

```bash
TMPDIR=/tmp CONDA_NO_PLUGINS=true PYTHONDONTWRITEBYTECODE=1 \
python -u \
  pitchRoll2UpperLower/t_p_v/train_ankle_pr_ul_relation_mlp.py \
  --csv pitchRoll2UpperLower/t_p_v/data/policy_pr_ul_walk.csv \
  --relation pr_to_ul_full \
  --save-dir pitchRoll2UpperLower/t_p_v/models_pr_ul_relation_mlp \
  --epochs 300 \
  --overwrite
```

Useful `--relation` choices:

```text
pr_to_ul_state   pitch/roll pos+vel -> upper/lower pos+vel
ul_to_pr_state   upper/lower pos+vel -> pitch/roll pos+vel
pr_to_ul_torque  pitch/roll pos+vel+tau -> upper/lower tau
ul_to_pr_torque  upper/lower pos+vel+tau -> pitch/roll tau
pr_to_ul_full    pitch/roll pos+vel+tau -> upper/lower pos+vel+tau
ul_to_pr_full    upper/lower pos+vel+tau -> pitch/roll pos+vel+tau
```
