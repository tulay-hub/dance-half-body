"""Collect suspended-base pitch/roll ankle data with measured and mapped torque.

The robot base is kinematically held in the air.  The script commands the
pitch/roll ankle joints with MuJoCo position-PD actuators, records measured
pitch/roll actuator torque, maps pitch/roll state to real upper/lower ankle
state, and writes two upper/lower torque estimates:

* tau_equiv: virtual-work mapping from measured PR torque,
  tau_pr = J_pr2ul.T @ tau_ul.
* tau_calc_model: the model-style calculation used by ankle_model
  train_effort_model*.py, tau_ul = J_pr2ul.T @ F_pr.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
import time
from pathlib import Path


def _wants_headless() -> bool:
    if "--headless" in sys.argv:
        return True
    for i, arg in enumerate(sys.argv):
        if arg == "--render_mode" and i + 1 < len(sys.argv):
            return sys.argv[i + 1] in {"none", "video"}
        if arg.startswith("--render_mode="):
            return arg.split("=", 1)[1] in {"none", "video"}
    return False


if _wants_headless():
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")

import cv2
import mujoco
import mujoco_viewer
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
PR_RUNNER_PATH = REPO_ROOT / "scripts" / "sim2sim_pr" / "lens110_pr_real_ankle.py"
DEFAULT_CONFIG = REPO_ROOT / "scripts" / "sim2sim_pr" / "pr_real_ankle.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data" / "suspended_pr_ankle_torque.csv"

ANKLE_NAMES = {
    "left_pitch": "left_ankle_pitch_joint",
    "left_roll": "left_ankle_roll_joint",
    "right_pitch": "right_ankle_pitch_joint",
    "right_roll": "right_ankle_roll_joint",
}
UL_NAMES = ("left_upper", "left_lower", "right_upper", "right_lower")


def load_pr_runner():
    spec = importlib.util.spec_from_file_location("lens110_pr_runner", PR_RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {PR_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


pr = load_pr_runner()


def mapper_copy(mapper) -> dict[str, np.ndarray]:
    return {side: value.copy() for side, value in mapper.previous.items()}


def restore_mapper(mapper, previous: dict[str, np.ndarray]) -> None:
    mapper.previous = {side: value.copy() for side, value in previous.items()}


def side_jacobian(mapper, pr_values: np.ndarray, side: str, eps: float) -> np.ndarray:
    if side == "left":
        pitch_index = mapper.observe_index["left_ankle_pitch_joint"]
        roll_index = mapper.observe_index["left_ankle_roll_joint"]
        out_slice = slice(0, 2)
    else:
        pitch_index = mapper.observe_index["right_ankle_pitch_joint"]
        roll_index = mapper.observe_index["right_ankle_roll_joint"]
        out_slice = slice(2, 4)

    saved = mapper_copy(mapper)
    cols = []
    for idx in (pitch_index, roll_index):
        plus = pr_values.copy()
        minus = pr_values.copy()
        plus[idx] += eps
        minus[idx] -= eps
        restore_mapper(mapper, saved)
        ul_plus = mapper.from_pitch_roll_targets(plus)[out_slice]
        restore_mapper(mapper, saved)
        ul_minus = mapper.from_pitch_roll_targets(minus)[out_slice]
        cols.append((ul_plus - ul_minus) / (2.0 * eps))
    restore_mapper(mapper, saved)
    return np.column_stack(cols)


def solve_equiv_ul_torque(j_pr2ul: np.ndarray, tau_pr: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(j_pr2ul.T, tau_pr)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(j_pr2ul.T, tau_pr, rcond=None)[0]


def fixed_root(data: mujoco.MjData, base_pos: np.ndarray, base_quat: np.ndarray) -> None:
    data.qpos[:3] = base_pos
    data.qpos[3:7] = base_quat
    data.qvel[:6] = 0.0


def init_render(model: mujoco.MjModel, data: mujoco.MjData, args: argparse.Namespace, cfg):
    if args.render_mode == "none":
        return None, None, None, None
    if args.render_mode == "video":
        model.vis.global_.offwidth = args.viewer_width
        model.vis.global_.offheight = args.viewer_height
        renderer = mujoco.Renderer(model, width=args.viewer_width, height=args.viewer_height)
        camera = mujoco.MjvCamera()
        camera.distance = args.camera_distance
        camera.azimuth = args.camera_azimuth
        camera.elevation = args.camera_elevation
        camera.lookat = [0.0, 0.0, args.base_z]
        writer = cv2.VideoWriter(
            str(args.video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            max(1.0, 1.0 / cfg.dt / args.render_every),
            (args.viewer_width, args.viewer_height),
        )
        return renderer, camera, writer, None

    viewer = mujoco_viewer.MujocoViewer(model, data, mode="window", width=args.viewer_width, height=args.viewer_height)
    viewer.cam.distance = args.camera_distance
    viewer.cam.azimuth = args.camera_azimuth
    viewer.cam.elevation = args.camera_elevation
    viewer.cam.lookat = [0.0, 0.0, args.base_z]
    return None, None, None, viewer


def render(data: mujoco.MjData, renderer, camera, writer, viewer, render_mode: str) -> bool:
    if render_mode == "none":
        return True
    if render_mode == "video":
        renderer.update_scene(data, camera=camera)
        writer.write(renderer.render())
        return True
    try:
        viewer.render()
    except Exception as exc:
        if "GLFW window does not exist" in str(exc):
            return False
        raise
    return True


def _smoothstep(x: float) -> tuple[float, float]:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x), 6.0 * x * (1.0 - x)


def _blend_value(t: float, boundary: float, width: float, low: float, high: float) -> tuple[float, float]:
    if width <= 0.0:
        return (high, 0.0) if t >= boundary else (low, 0.0)
    start = boundary - 0.5 * width
    x = (t - start) / width
    s, ds_dx = _smoothstep(x)
    return low + (high - low) * s, (high - low) * ds_dx / width


def scheduled_amplitudes(t: float, args: argparse.Namespace) -> tuple[float, float, float, float, str]:
    if args.amp_schedule == "constant":
        return args.pitch_amp, args.roll_amp, 0.0, 0.0, "constant"

    total = max(float(args.sim_duration), 1e-6)
    b1 = total / 3.0
    b2 = 2.0 * total / 3.0
    width = min(max(args.stage_blend_time, 0.0), 0.45 * total / 3.0)

    p_small = args.pitch_amp_small if args.pitch_amp_small is not None else args.pitch_amp
    r_small = args.roll_amp_small if args.roll_amp_small is not None else args.roll_amp
    p_mid = args.pitch_amp_mid if args.pitch_amp_mid is not None else 0.52
    r_mid = args.roll_amp_mid if args.roll_amp_mid is not None else 0.34
    p_max = args.pitch_amp_max if args.pitch_amp_max is not None else 0.78
    r_max = args.roll_amp_max if args.roll_amp_max is not None else 0.46

    if t < b1 + 0.5 * width:
        p, dp = _blend_value(t, b1, width, p_small, p_mid)
        r, dr = _blend_value(t, b1, width, r_small, r_mid)
        stage = "small" if t < b1 - 0.5 * width else "small_to_mid"
    elif t < b2 + 0.5 * width:
        p, dp = _blend_value(t, b2, width, p_mid, p_max)
        r, dr = _blend_value(t, b2, width, r_mid, r_max)
        stage = "mid" if t < b2 - 0.5 * width else "mid_to_max"
    else:
        p, r, dp, dr, stage = p_max, r_max, 0.0, 0.0, "max"
    return float(p), float(r), float(dp), float(dr), stage


def target_signal(t: float, args: argparse.Namespace) -> tuple[dict[str, float], dict[str, float]]:
    f = args.freq
    ap, ar, dap, dar, stage = scheduled_amplitudes(t, args)
    args.current_pitch_amp = ap
    args.current_roll_amp = ar
    args.current_amp_stage = stage
    cp = args.pitch_center
    cr = args.roll_center

    if args.motion_pattern == "circle":
        w = 2.0 * np.pi * f
        left_theta = w * t + args.left_circle_phase
        right_theta = w * t + args.left_circle_phase + args.right_circle_phase_offset
        specs = {
            "left_pitch": (cp, ap, dap, left_theta, "sin", w),
            "left_roll": (cr, ar, dar, left_theta, "cos", w),
            "right_pitch": (cp, ap, dap, right_theta, "sin", w),
            "right_roll": (cr, ar, dar, right_theta, "cos", w),
        }
        pos = {}
        vel = {}
        for name, (center, amp, amp_dot, theta, kind, omega) in specs.items():
            if kind == "sin":
                wave = np.sin(theta)
                wave_dot = omega * np.cos(theta)
            else:
                wave = np.cos(theta)
                wave_dot = -omega * np.sin(theta)
            pos[name] = center + amp * wave
            vel[name] = amp_dot * wave + amp * wave_dot
        return pos, vel

    specs = {
        "left_pitch": (cp, ap, dap, f, 0.00, 0.31, 1.73 * f, 0.40),
        "left_roll": (cr, ar, dar, 1.31 * f, 0.37, 0.27, 2.11 * f, 1.10),
        "right_pitch": (cp, ap, dap, 0.83 * f, 1.27, 0.29, 1.91 * f, 2.20),
        "right_roll": (cr, ar, dar, 1.57 * f, 2.41, 0.23, 2.37 * f, 0.80),
    }
    pos = {}
    vel = {}
    for name, (center, amp, amp_dot, f1, ph1, mix, f2, ph2) in specs.items():
        w1 = 2.0 * np.pi * f1
        w2 = 2.0 * np.pi * f2
        wave = np.sin(w1 * t + ph1) + mix * np.sin(w2 * t + ph2)
        wave_dot = w1 * np.cos(w1 * t + ph1) + mix * w2 * np.cos(w2 * t + ph2)
        pos[name] = center + amp * wave
        vel[name] = amp_dot * wave + amp * wave_dot
    return pos, vel


def apply_ankle_targets(target_pos: np.ndarray, target_vel: np.ndarray, robot, args: argparse.Namespace) -> None:
    pos_cmd, vel_cmd = target_signal(args.current_time, args)
    for short_name, joint_name in ANKLE_NAMES.items():
        idx = robot.name_to_mujoco[joint_name]
        lo = robot.target_min[idx]
        hi = robot.target_max[idx]
        unclipped = pos_cmd[short_name]
        clipped = float(np.clip(unclipped, lo, hi))
        target_pos[idx] = clipped
        target_vel[idx] = 0.0 if clipped != unclipped else vel_cmd[short_name]


def make_fieldnames() -> list[str]:
    fields = [
        "step",
        "time",
        "root_x",
        "root_y",
        "root_z",
        "contact_count",
        "amp_stage",
        "motion_pattern",
        "pitch_amp_current",
        "roll_amp_current",
    ]
    for name in ("left_pitch", "left_roll", "right_pitch", "right_roll"):
        fields.extend(
            [
                f"{name}_pos",
                f"{name}_vel",
                f"{name}_target",
                f"{name}_target_vel",
                f"{name}_error",
                f"{name}_tau",
                f"{name}_tau_mj",
                f"{name}_tau_pd",
                f"{name}_tau_pd_unclipped",
            ]
        )
    for name in UL_NAMES:
        fields.extend(
            [
                f"{name}_pos",
                f"{name}_vel",
                f"{name}_target",
                f"{name}_target_vel",
                f"{name}_error",
                f"{name}_tau",
                f"{name}_tau_equiv",
                f"{name}_tau_equiv_from_pd",
                f"{name}_tau_calc_model",
                f"{name}_tau_calc_model_unclipped",
            ]
        )
    fields.extend(
        [
            "left_tau_equiv_residual",
            "right_tau_equiv_residual",
            "left_tau_equiv_pd_residual",
            "right_tau_equiv_pd_residual",
            "left_jacobian_cond",
            "right_jacobian_cond",
            "source",
        ]
    )
    return fields


def side_values(robot, values: np.ndarray, side: str) -> np.ndarray:
    if side == "left":
        return values[
            [
                robot.name_to_mujoco["left_ankle_pitch_joint"],
                robot.name_to_mujoco["left_ankle_roll_joint"],
            ]
        ]
    return values[
        [
            robot.name_to_mujoco["right_ankle_pitch_joint"],
            robot.name_to_mujoco["right_ankle_roll_joint"],
        ]
    ]


def fill_pr_columns(row: dict, robot, cfg, q, dq, target_pos, target_vel, tau_mj, tau_pd, tau_pd_unclipped) -> None:
    for short_name, joint_name in ANKLE_NAMES.items():
        idx = robot.name_to_mujoco[joint_name]
        row[f"{short_name}_pos"] = float(q[idx])
        row[f"{short_name}_vel"] = float(dq[idx])
        row[f"{short_name}_target"] = float(target_pos[idx])
        row[f"{short_name}_target_vel"] = float(target_vel[idx])
        row[f"{short_name}_error"] = float(target_pos[idx] - q[idx])
        row[f"{short_name}_tau"] = float(tau_mj[idx])
        row[f"{short_name}_tau_mj"] = float(tau_mj[idx])
        row[f"{short_name}_tau_pd"] = float(tau_pd[idx])
        row[f"{short_name}_tau_pd_unclipped"] = float(tau_pd_unclipped[idx])


def fill_ul_columns(
    row: dict,
    ul_pos: np.ndarray,
    ul_vel: np.ndarray,
    ul_target: np.ndarray,
    ul_target_vel: np.ndarray,
    ul_tau_equiv: np.ndarray,
    ul_tau_equiv_pd: np.ndarray,
    ul_tau_calc_model: np.ndarray,
    ul_tau_calc_model_unclipped: np.ndarray,
) -> None:
    for i, name in enumerate(UL_NAMES):
        row[f"{name}_pos"] = float(ul_pos[i])
        row[f"{name}_vel"] = float(ul_vel[i])
        row[f"{name}_target"] = float(ul_target[i])
        row[f"{name}_target_vel"] = float(ul_target_vel[i])
        row[f"{name}_error"] = float(ul_target[i] - ul_pos[i])
        row[f"{name}_tau"] = float(ul_tau_equiv[i])
        row[f"{name}_tau_equiv"] = float(ul_tau_equiv[i])
        row[f"{name}_tau_equiv_from_pd"] = float(ul_tau_equiv_pd[i])
        row[f"{name}_tau_calc_model"] = float(ul_tau_calc_model[i])
        row[f"{name}_tau_calc_model_unclipped"] = float(ul_tau_calc_model_unclipped[i])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect suspended-base PR ankle torque data.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--real_ankle_model_path", default=None)
    parser.add_argument("--output_csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sim_duration", type=float, default=30.0)
    parser.add_argument("--sample_every", type=int, default=1)
    parser.add_argument("--render_mode", choices=("none", "window", "video"), default="none")
    parser.add_argument("--headless", action="store_true", help="Alias for --render_mode none.")
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--render_every", type=int, default=4)
    parser.add_argument("--video_path", type=Path, default=Path(__file__).resolve().parent / "data" / "suspended_pr_ankle_torque.mp4")
    parser.add_argument("--viewer_width", type=int, default=1280)
    parser.add_argument("--viewer_height", type=int, default=720)
    parser.add_argument("--camera_distance", type=float, default=3.2)
    parser.add_argument("--camera_azimuth", type=float, default=90.0)
    parser.add_argument("--camera_elevation", type=float, default=-18.0)
    parser.add_argument("--base_z", type=float, default=1.05)
    parser.add_argument("--pitch_center", type=float, default=-0.15)
    parser.add_argument("--roll_center", type=float, default=0.0)
    parser.add_argument("--pitch_amp", type=float, default=0.28)
    parser.add_argument("--roll_amp", type=float, default=0.22)
    parser.add_argument("--amp_schedule", choices=("constant", "staged"), default="constant")
    parser.add_argument("--pitch_amp_small", type=float, default=None)
    parser.add_argument("--roll_amp_small", type=float, default=None)
    parser.add_argument("--pitch_amp_mid", type=float, default=None)
    parser.add_argument("--roll_amp_mid", type=float, default=None)
    parser.add_argument("--pitch_amp_max", type=float, default=None)
    parser.add_argument("--roll_amp_max", type=float, default=None)
    parser.add_argument("--stage_blend_time", type=float, default=2.0)
    parser.add_argument("--motion_pattern", choices=("multisine", "circle"), default="multisine")
    parser.add_argument("--left_circle_phase", type=float, default=0.0)
    parser.add_argument("--right_circle_phase_offset", type=float, default=3.141592653589793)
    parser.add_argument("--freq", type=float, default=0.35)
    parser.add_argument("--ankle_kp_scale", type=float, default=1.0)
    parser.add_argument("--ankle_kd_scale", type=float, default=1.0)
    parser.add_argument("--jacobian_eps", type=float, default=1e-5)
    parser.add_argument("--print_every", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.headless:
        args.render_mode = "none"

    cfg = pr.load_config(str(args.config))
    args.model_path = pr.resolve_path(args.model_path) if args.model_path else cfg.model_path
    args.real_ankle_model_path = (
        pr.resolve_path(args.real_ankle_model_path) if args.real_ankle_model_path else cfg.real_ankle_model_path
    )

    model = mujoco.MjModel.from_xml_path(args.model_path)
    model.opt.timestep = cfg.dt
    data = mujoco.MjData(model)

    robot = pr.RobotRuntime(
        policy_to_mujoco=pr.make_index_mapping(cfg.policy_joint_names, cfg.mujoco_joint_names),
        default_pos=cfg.default_pos.copy(),
        action_scale=cfg.action_scale.copy(),
        base_pos=np.array([0.0, 0.0, args.base_z], dtype=np.float64),
        action_clip=cfg.action_clip,
        static_friction=cfg.static_friction,
        dynamic_friction=cfg.dynamic_friction,
    )
    robot.qpos_ids, robot.qvel_ids, robot.actuator_ids, robot.target_min, robot.target_max = pr.build_joint_data(
        model, cfg
    )
    robot.name_to_mujoco = {name: i for i, name in enumerate(cfg.mujoco_joint_names)}

    cfg.kp = cfg.kp.copy()
    cfg.kd = cfg.kd.copy()
    for joint_name in ANKLE_NAMES.values():
        idx = robot.name_to_mujoco[joint_name]
        cfg.kp[idx] *= args.ankle_kp_scale
        cfg.kd[idx] *= args.ankle_kd_scale
    pr.patch_model(model, robot, cfg)

    state_mapper = pr.RealAnkleCommandMapper(args.real_ankle_model_path, cfg)
    target_mapper = pr.RealAnkleCommandMapper(args.real_ankle_model_path, cfg)
    jacobian_mapper = pr.RealAnkleCommandMapper(args.real_ankle_model_path, cfg)

    base_pos = np.array([0.0, 0.0, args.base_z], dtype=np.float64)
    base_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    data.qpos[:3] = base_pos
    data.qpos[3:7] = base_quat
    data.qpos[robot.qpos_ids] = robot.default_pos
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    data.ctrl[robot.actuator_ids] = robot.default_pos
    fixed_root(data, base_pos, base_quat)
    mujoco.mj_forward(model, data)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    if args.output_csv.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {args.output_csv}. Use --overwrite.")
    if args.render_mode == "video":
        args.video_path.parent.mkdir(parents=True, exist_ok=True)

    renderer, camera, video_writer, viewer = init_render(model, data, args, cfg)

    target_pos = robot.default_pos.copy()
    target_vel = np.zeros(cfg.num_actions, dtype=np.float64)
    total_steps = int(args.sim_duration / cfg.dt)
    sample_every = max(1, int(args.sample_every))
    rows_written = 0
    max_abs_equiv_residual = 0.0
    start = time.time()

    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=make_fieldnames())
        writer.writeheader()

        for step in range(total_steps):
            t = step * cfg.dt
            args.current_time = t
            fixed_root(data, base_pos, base_quat)
            target_pos[:] = robot.default_pos
            target_vel[:] = 0.0
            apply_ankle_targets(target_pos, target_vel, robot, args)
            data.ctrl[robot.actuator_ids] = target_pos

            mujoco.mj_step(model, data)
            fixed_root(data, base_pos, base_quat)
            mujoco.mj_forward(model, data)

            if step % sample_every != 0:
                if args.render_mode != "none" and step % max(1, args.render_every) == 0:
                    if not render(data, renderer, camera, video_writer, viewer, args.render_mode):
                        print("[INFO] viewer closed; stopping collection")
                        break
                if args.realtime:
                    elapsed = time.time() - start
                    target_time = (step + 1) * cfg.dt
                    if elapsed < target_time:
                        time.sleep(target_time - elapsed)
                continue

            q = data.qpos[robot.qpos_ids].copy()
            dq = data.qvel[robot.qvel_ids].copy()
            tau_mj = data.actuator_force[robot.actuator_ids].copy()
            tau_pd_unclipped = cfg.kp * (target_pos - q) - cfg.kd * dq
            tau_pd = np.clip(tau_pd_unclipped, -cfg.tau_limit, cfg.tau_limit)

            ul_pos = state_mapper.from_pitch_roll_targets(q)
            ul_target = target_mapper.from_pitch_roll_targets(target_pos)
            left_j = side_jacobian(jacobian_mapper, q, "left", args.jacobian_eps)
            right_j = side_jacobian(jacobian_mapper, q, "right", args.jacobian_eps)
            left_j_target = side_jacobian(jacobian_mapper, target_pos, "left", args.jacobian_eps)
            right_j_target = side_jacobian(jacobian_mapper, target_pos, "right", args.jacobian_eps)

            left_pr_vel = side_values(robot, dq, "left")
            right_pr_vel = side_values(robot, dq, "right")
            left_pr_target_vel = side_values(robot, target_vel, "left")
            right_pr_target_vel = side_values(robot, target_vel, "right")
            ul_vel = np.concatenate([left_j @ left_pr_vel, right_j @ right_pr_vel])
            ul_target_vel = np.concatenate([left_j_target @ left_pr_target_vel, right_j_target @ right_pr_target_vel])

            tau_pr_mj_left = side_values(robot, tau_mj, "left")
            tau_pr_mj_right = side_values(robot, tau_mj, "right")
            tau_pr_pd_left = side_values(robot, tau_pd, "left")
            tau_pr_pd_right = side_values(robot, tau_pd, "right")
            tau_pr_pd_unclipped_left = side_values(robot, tau_pd_unclipped, "left")
            tau_pr_pd_unclipped_right = side_values(robot, tau_pd_unclipped, "right")

            tau_ul_equiv_left = solve_equiv_ul_torque(left_j, tau_pr_mj_left)
            tau_ul_equiv_right = solve_equiv_ul_torque(right_j, tau_pr_mj_right)
            tau_ul_equiv_pd_left = solve_equiv_ul_torque(left_j, tau_pr_pd_left)
            tau_ul_equiv_pd_right = solve_equiv_ul_torque(right_j, tau_pr_pd_right)
            tau_ul_calc_left = left_j.T @ tau_pr_pd_left
            tau_ul_calc_right = right_j.T @ tau_pr_pd_right
            tau_ul_calc_unclipped_left = left_j.T @ tau_pr_pd_unclipped_left
            tau_ul_calc_unclipped_right = right_j.T @ tau_pr_pd_unclipped_right

            ul_tau_equiv = np.concatenate([tau_ul_equiv_left, tau_ul_equiv_right])
            ul_tau_equiv_pd = np.concatenate([tau_ul_equiv_pd_left, tau_ul_equiv_pd_right])
            ul_tau_calc = np.concatenate([tau_ul_calc_left, tau_ul_calc_right])
            ul_tau_calc_unclipped = np.concatenate([tau_ul_calc_unclipped_left, tau_ul_calc_unclipped_right])

            left_resid = float(np.linalg.norm(left_j.T @ tau_ul_equiv_left - tau_pr_mj_left))
            right_resid = float(np.linalg.norm(right_j.T @ tau_ul_equiv_right - tau_pr_mj_right))
            left_resid_pd = float(np.linalg.norm(left_j.T @ tau_ul_equiv_pd_left - tau_pr_pd_left))
            right_resid_pd = float(np.linalg.norm(right_j.T @ tau_ul_equiv_pd_right - tau_pr_pd_right))
            max_abs_equiv_residual = max(max_abs_equiv_residual, left_resid, right_resid, left_resid_pd, right_resid_pd)

            row = {
                "step": int(step),
                "time": float(data.time),
                "root_x": float(data.qpos[0]),
                "root_y": float(data.qpos[1]),
                "root_z": float(data.qpos[2]),
                "contact_count": int(data.ncon),
                "amp_stage": getattr(args, "current_amp_stage", "constant"),
                "motion_pattern": args.motion_pattern,
                "pitch_amp_current": float(getattr(args, "current_pitch_amp", args.pitch_amp)),
                "roll_amp_current": float(getattr(args, "current_roll_amp", args.roll_amp)),
                "left_tau_equiv_residual": left_resid,
                "right_tau_equiv_residual": right_resid,
                "left_tau_equiv_pd_residual": left_resid_pd,
                "right_tau_equiv_pd_residual": right_resid_pd,
                "left_jacobian_cond": float(np.linalg.cond(left_j)),
                "right_jacobian_cond": float(np.linalg.cond(right_j)),
                "source": "suspended_pr_ankle_pitch_roll_pd",
            }
            fill_pr_columns(row, robot, cfg, q, dq, target_pos, target_vel, tau_mj, tau_pd, tau_pd_unclipped)
            fill_ul_columns(
                row,
                ul_pos,
                ul_vel,
                ul_target,
                ul_target_vel,
                ul_tau_equiv,
                ul_tau_equiv_pd,
                ul_tau_calc,
                ul_tau_calc_unclipped,
            )
            writer.writerow(row)
            rows_written += 1

            if step % max(1, args.print_every) == 0:
                print(
                    f"step={step:06d} rows={rows_written} "
                    f"lp={q[robot.name_to_mujoco['left_ankle_pitch_joint']]:+.3f} "
                    f"lr={q[robot.name_to_mujoco['left_ankle_roll_joint']]:+.3f} "
                    f"rp={q[robot.name_to_mujoco['right_ankle_pitch_joint']]:+.3f} "
                    f"rr={q[robot.name_to_mujoco['right_ankle_roll_joint']]:+.3f} "
                    f"pattern={args.motion_pattern} "
                    f"stage={getattr(args, 'current_amp_stage', 'constant')} "
                    f"amp=({getattr(args, 'current_pitch_amp', args.pitch_amp):.2f},"
                    f"{getattr(args, 'current_roll_amp', args.roll_amp):.2f}) "
                    f"max_resid={max_abs_equiv_residual:.2e}"
                )

            if args.render_mode != "none" and step % max(1, args.render_every) == 0:
                if not render(data, renderer, camera, video_writer, viewer, args.render_mode):
                    print("[INFO] viewer closed; stopping collection")
                    break

            if args.realtime:
                elapsed = time.time() - start
                target_time = (step + 1) * cfg.dt
                if elapsed < target_time:
                    time.sleep(target_time - elapsed)

    if args.render_mode == "video":
        video_writer.release()
        print(f"[INFO] saved video: {args.video_path}")
    elif args.render_mode == "window" and viewer is not None:
        viewer.close()

    print(
        f"[SUMMARY] wrote {rows_written} rows: {args.output_csv}; "
        f"max_equiv_residual={max_abs_equiv_residual:.3e}"
    )


if __name__ == "__main__":
    main()
