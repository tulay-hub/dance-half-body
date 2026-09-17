"""Collect pitch/roll and equivalent upper/lower ankle data from a stable policy.

This uses the trained pitch/roll ankle policy runner as the source of motion.
The MuJoCo simulation remains on the stable 21DoF pitch/roll model; upper/lower
ankle position and velocity are mapped through the real Lens110 ankle tendon
geometry, and upper/lower torque is reported as the virtual-work equivalent.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np


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

import mujoco
try:
    import cv2
except ImportError:  # Video output is optional for headless data collection.
    cv2 = None
try:
    import mujoco_viewer
except ImportError:  # Window rendering is optional for headless data collection.
    mujoco_viewer = None


REPO_ROOT = Path(__file__).resolve().parents[2]
PR_RUNNER_PATH = REPO_ROOT / "scripts" / "sim2sim_pr" / "lens110_pr_real_ankle.py"
DEFAULT_CONFIG = REPO_ROOT / "scripts" / "sim2sim_pr" / "pr_real_ankle.json"
DEFAULT_POLICY = (
    REPO_ROOT / "logs" / "rsl_rl" / "lens110_amp" / "2026-06-26_18-04-38" / "exported" / "policy.onnx"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data" / "policy_pr_ul.csv"

ANKLE_JOINTS = {
    "left_pitch": "left_ankle_pitch_joint",
    "left_roll": "left_ankle_roll_joint",
    "right_pitch": "right_ankle_pitch_joint",
    "right_roll": "right_ankle_roll_joint",
}
UL_NAMES = ("left_upper", "left_lower", "right_upper", "right_lower")


def load_pr_runner():
    spec = importlib.util.spec_from_file_location("lens110_pr_real_ankle_runner", PR_RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {PR_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


pr = load_pr_runner()


def init_render(model: mujoco.MjModel, data: mujoco.MjData, args: argparse.Namespace, cfg):
    if args.render_mode == "none":
        return None, None, None, None

    if args.render_mode == "video":
        if cv2 is None:
            raise ImportError("Video rendering requires OpenCV; install opencv-python in the active local environment.")
        model.vis.global_.offwidth = args.viewer_width
        model.vis.global_.offheight = args.viewer_height
        renderer = mujoco.Renderer(model, width=args.viewer_width, height=args.viewer_height)
        camera = mujoco.MjvCamera()
        camera.distance = args.camera_distance
        camera.azimuth = args.camera_azimuth
        camera.elevation = args.camera_elevation
        camera.lookat = [0.0, 0.0, 1.0]
        writer = cv2.VideoWriter(
            str(args.video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            1.0 / cfg.dt / cfg.decimation,
            (args.viewer_width, args.viewer_height),
        )
        return renderer, camera, writer, None

    if mujoco_viewer is None:
        raise ImportError("Window rendering requires mujoco-python-viewer in the active local environment.")
    viewer = mujoco_viewer.MujocoViewer(model, data, mode="window", width=args.viewer_width, height=args.viewer_height)
    viewer.cam.distance = args.camera_distance
    viewer.cam.azimuth = args.camera_azimuth
    viewer.cam.elevation = args.camera_elevation
    viewer.cam.lookat = [0.0, 0.0, 1.0]
    return None, None, None, viewer


def render(data: mujoco.MjData, renderer, camera, writer, viewer, render_mode: str) -> bool:
    if render_mode == "none":
        return True
    if pr.CommandState.camera_follow:
        base_pos = [float(x) for x in data.qpos[:3]]
        if render_mode == "video":
            camera.lookat = base_pos
        else:
            viewer.cam.lookat = base_pos
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


def mapper_copy(mapper) -> dict[str, np.ndarray]:
    return {side: value.copy() for side, value in mapper.previous.items()}


def restore_mapper(mapper, previous: dict[str, np.ndarray]) -> None:
    mapper.previous = {side: value.copy() for side, value in previous.items()}


def map_pr_to_ul(mapper, pr_values: np.ndarray) -> np.ndarray:
    return mapper.from_pitch_roll_targets(pr_values)


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
        ul_plus = map_pr_to_ul(mapper, plus)[out_slice]
        restore_mapper(mapper, saved)
        ul_minus = map_pr_to_ul(mapper, minus)[out_slice]
        cols.append((ul_plus - ul_minus) / (2.0 * eps))
    restore_mapper(mapper, saved)
    return np.column_stack(cols)


def equivalent_ul_torque(jacobian: np.ndarray, tau_pr: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(jacobian.T, tau_pr)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(jacobian.T, tau_pr, rcond=None)[0]


def make_fieldnames() -> list[str]:
    fields = [
        "step",
        "policy_step",
        "time",
        "cmd_vx",
        "cmd_vy",
        "cmd_yaw",
        "base_vx",
        "base_vy",
        "base_yaw_rate",
        "root_z",
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
            ]
        )
    fields.extend(
        [
            "left_tau_equiv_residual",
            "right_tau_equiv_residual",
            "left_jacobian_cond",
            "right_jacobian_cond",
            "contact_count",
            "source",
        ]
    )
    return fields


def row_from_state(
    step: int,
    cfg,
    robot,
    data: mujoco.MjData,
    lin_vel_b: np.ndarray,
    omega_b: np.ndarray,
    q: np.ndarray,
    dq: np.ndarray,
    target_pos: np.ndarray,
    prev_target_pos: np.ndarray,
    ul_pos: np.ndarray,
    ul_target: np.ndarray,
    prev_ul_pos: np.ndarray,
    prev_ul_target: np.ndarray,
    tau: np.ndarray,
    ul_tau: np.ndarray,
    left_residual: float,
    right_residual: float,
    left_cond: float,
    right_cond: float,
) -> dict[str, float | int | str]:
    dt_policy = cfg.dt * cfg.decimation
    row: dict[str, float | int | str] = {
        "step": int(step),
        "policy_step": int(step // cfg.decimation),
        "time": float(data.time),
        "cmd_vx": float(pr.CommandState.vx),
        "cmd_vy": float(pr.CommandState.vy),
        "cmd_yaw": float(pr.CommandState.dyaw),
        "base_vx": float(lin_vel_b[0]),
        "base_vy": float(lin_vel_b[1]),
        "base_yaw_rate": float(omega_b[2]),
        "root_z": float(data.qpos[2]),
        "left_tau_equiv_residual": float(left_residual),
        "right_tau_equiv_residual": float(right_residual),
        "left_jacobian_cond": float(left_cond),
        "right_jacobian_cond": float(right_cond),
        "contact_count": int(data.ncon),
        "source": "policy_pr_real_ankle_equivalent_ul",
    }
    for name, joint_name in ANKLE_JOINTS.items():
        idx = robot.name_to_mujoco[joint_name]
        target_vel = (target_pos[idx] - prev_target_pos[idx]) / dt_policy
        row[f"{name}_pos"] = float(q[idx])
        row[f"{name}_vel"] = float(dq[idx])
        row[f"{name}_target"] = float(target_pos[idx])
        row[f"{name}_target_vel"] = float(target_vel)
        row[f"{name}_error"] = float(target_pos[idx] - q[idx])
        row[f"{name}_tau"] = float(tau[idx])
    for i, name in enumerate(UL_NAMES):
        row[f"{name}_pos"] = float(ul_pos[i])
        row[f"{name}_vel"] = float((ul_pos[i] - prev_ul_pos[i]) / dt_policy)
        row[f"{name}_target"] = float(ul_target[i])
        row[f"{name}_target_vel"] = float((ul_target[i] - prev_ul_target[i]) / dt_policy)
        row[f"{name}_error"] = float(ul_target[i] - ul_pos[i])
        row[f"{name}_tau"] = float(ul_tau[i])
    return row


def prepare(args: argparse.Namespace):
    cfg = pr.load_config(str(args.config))
    args.load_model = pr.resolve_path(args.load_model) if args.load_model else cfg.torchscript_policy
    args.load_onnx = pr.resolve_path(args.load_onnx) if args.load_onnx else cfg.onnx_policy
    args.policy_backend = args.policy_backend or cfg.policy_backend
    args.model_path = pr.resolve_path(args.model_path) if args.model_path else cfg.model_path
    args.real_ankle_model_path = (
        pr.resolve_path(args.real_ankle_model_path) if args.real_ankle_model_path else cfg.real_ankle_model_path
    )
    args.init_height = args.init_height if args.init_height is not None else (cfg.init_height or cfg.fallback_init_height)
    if args.load_onnx is None and args.policy_backend == "onnx":
        args.load_onnx = str(DEFAULT_POLICY)
    return cfg


def run(args: argparse.Namespace) -> None:
    if args.headless:
        args.render_mode = "none"
    cfg = prepare(args)
    pr.CommandState.set_initial(tuple(args.cmd_vel))

    if args.policy_backend == "onnx":
        policy = pr.OnnxPolicy(args.load_onnx)
        print(f"[INFO] loaded ONNX policy with {policy.backend}: {args.load_onnx}")
    else:
        policy = pr.TorchScriptPolicy(args.load_model)
        print(f"[INFO] loaded TorchScript policy: {args.load_model}")

    print("Keyboard: 8/2 vx, 4/6 vy, 7/9 yaw, arrows vx/yaw, 0 reset, F camera follow")
    listener = pr.start_keyboard_listener() if args.render_mode == "window" else None

    model = mujoco.MjModel.from_xml_path(args.model_path)
    model.opt.timestep = cfg.dt
    data = mujoco.MjData(model)
    target_mapper = pr.RealAnkleCommandMapper(args.real_ankle_model_path, cfg)
    state_mapper = pr.RealAnkleCommandMapper(args.real_ankle_model_path, cfg)
    jacobian_mapper = pr.RealAnkleCommandMapper(args.real_ankle_model_path, cfg)

    robot = pr.make_robot(args, cfg)
    robot.qpos_ids, robot.qvel_ids, robot.actuator_ids, robot.target_min, robot.target_max = pr.build_joint_data(
        model, cfg
    )
    robot.name_to_mujoco = {name: i for i, name in enumerate(cfg.mujoco_joint_names)}
    pr.patch_model(model, robot, cfg)

    if not args.no_auto_base_height:
        robot.base_pos[2] = pr.grounded_base_height(model, data, robot, cfg)
        print(f"[INFO] auto base height: {robot.base_pos[2]:.3f}")

    pr.hold_default_stand(data, robot)
    mujoco.mj_forward(model, data)
    initial_qpos = data.qpos.copy()
    initial_qvel = data.qvel.copy()

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    if args.output_csv.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {args.output_csv}. Use --overwrite.")
    if args.render_mode == "video":
        args.video_path.parent.mkdir(parents=True, exist_ok=True)

    renderer, camera, writer, viewer = init_render(model, data, args, cfg)

    action = np.zeros(cfg.num_actions, dtype=np.float64)
    target_pos = robot.default_pos.copy()
    prev_target_pos = target_pos.copy()
    q0 = data.qpos[robot.qpos_ids].copy()
    ul_target = target_mapper.from_pitch_roll_targets(target_pos)
    ul_pos = state_mapper.from_pitch_roll_targets(q0)
    prev_ul_target = ul_target.copy()
    prev_ul_pos = ul_pos.copy()

    fieldnames = make_fieldnames()
    rows_written = 0
    min_root_z = float("inf")
    start_time = time.time()
    total_steps = int(args.sim_duration / cfg.dt)
    sample_every = max(1, int(args.sample_every))

    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
        writer_csv.writeheader()

        for step in range(total_steps):
            if pr.CommandState.reset_requested:
                data.qpos[:] = initial_qpos
                data.qvel[:] = initial_qvel
                data.ctrl[:] = 0.0
                data.ctrl[robot.actuator_ids] = robot.default_pos
                action[:] = 0.0
                target_pos[:] = robot.default_pos
                prev_target_pos[:] = target_pos
                ul_target[:] = target_mapper.from_pitch_roll_targets(target_pos)
                ul_pos[:] = state_mapper.from_pitch_roll_targets(target_pos)
                prev_ul_target[:] = ul_target
                prev_ul_pos[:] = ul_pos
                pr.CommandState.zero()
                pr.CommandState.reset_requested = False
                mujoco.mj_forward(model, data)

            qpos, qvel, lin_vel_b, omega_b, gravity_b = pr.get_base_obs(data)
            q = qpos[robot.qpos_ids].copy()
            dq = qvel[robot.qvel_ids].copy()
            min_root_z = min(min_root_z, float(qpos[2]))

            if step % cfg.decimation == 0:
                prev_target_pos = target_pos.copy()
                prev_ul_target = ul_target.copy()
                prev_ul_pos = ul_pos.copy()

                obs = pr.make_observation(q, dq, omega_b, gravity_b, action, robot, cfg)
                action[:] = policy(obs)
                if robot.action_clip is not None:
                    np.clip(action, -robot.action_clip, robot.action_clip, out=action)
                target_pos = pr.action_to_target(action, robot)
                ul_target = target_mapper.from_pitch_roll_targets(target_pos)
                ul_pos = state_mapper.from_pitch_roll_targets(q)

                left_pitch_i = robot.name_to_mujoco["left_ankle_pitch_joint"]
                left_roll_i = robot.name_to_mujoco["left_ankle_roll_joint"]
                right_pitch_i = robot.name_to_mujoco["right_ankle_pitch_joint"]
                right_roll_i = robot.name_to_mujoco["right_ankle_roll_joint"]
                tau = data.actuator_force[robot.actuator_ids].copy()

                left_j = side_jacobian(jacobian_mapper, q, "left", args.jacobian_eps)
                right_j = side_jacobian(jacobian_mapper, q, "right", args.jacobian_eps)
                tau_pr_left = tau[[left_pitch_i, left_roll_i]]
                tau_pr_right = tau[[right_pitch_i, right_roll_i]]
                tau_ul_left = equivalent_ul_torque(left_j, tau_pr_left)
                tau_ul_right = equivalent_ul_torque(right_j, tau_pr_right)
                ul_tau = np.array([tau_ul_left[0], tau_ul_left[1], tau_ul_right[0], tau_ul_right[1]], dtype=np.float64)
                left_residual = float(np.linalg.norm(left_j.T @ tau_ul_left - tau_pr_left))
                right_residual = float(np.linalg.norm(right_j.T @ tau_ul_right - tau_pr_right))
                left_cond = float(np.linalg.cond(left_j))
                right_cond = float(np.linalg.cond(right_j))

                policy_step = step // cfg.decimation
                if policy_step % sample_every == 0:
                    row = row_from_state(
                        step,
                        cfg,
                        robot,
                        data,
                        lin_vel_b,
                        omega_b,
                        q,
                        dq,
                        target_pos,
                        prev_target_pos,
                        ul_pos,
                        ul_target,
                        prev_ul_pos,
                        prev_ul_target,
                        tau,
                        ul_tau,
                        left_residual,
                        right_residual,
                        left_cond,
                        right_cond,
                    )
                    writer_csv.writerow(row)
                    rows_written += 1

                if policy_step % max(1, args.print_every) == 0:
                    print(
                        f"step={policy_step:06d} rows={rows_written} "
                        f"cmd=({pr.CommandState.vx:.2f}, {pr.CommandState.vy:.2f}, {pr.CommandState.dyaw:.2f}) "
                        f"vel=({lin_vel_b[0]:.2f}, {lin_vel_b[1]:.2f}, {omega_b[2]:.2f}) "
                        f"ul_target=({ul_target[0]:+.3f}, {ul_target[1]:+.3f}, "
                        f"{ul_target[2]:+.3f}, {ul_target[3]:+.3f})"
                    )

                if step % max(cfg.decimation, args.render_every * cfg.decimation) == 0:
                    if not render(data, renderer, camera, writer, viewer, args.render_mode):
                        print("[INFO] viewer closed; stopping collection")
                        break

            data.ctrl[robot.actuator_ids] = target_pos
            mujoco.mj_step(model, data)

            if args.realtime:
                elapsed = time.time() - start_time
                target_time = (step + 1) * cfg.dt
                if elapsed < target_time:
                    time.sleep(target_time - elapsed)

    if args.render_mode == "video":
        writer.release()
        print(f"[INFO] saved video: {args.video_path}")
    elif args.render_mode == "window" and viewer is not None:
        viewer.close()
    if listener is not None:
        listener.stop()

    print(
        f"[SUMMARY] wrote {rows_written} rows: {args.output_csv}; "
        f"min_root_z={min_root_z:.3f}; final_contacts={pr.summarize_contacts(model, data)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect PR and equivalent UL ankle data from a trained PR policy.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--load_model", default=None, help="Override TorchScript policy path from config.")
    parser.add_argument("--load_onnx", default=str(DEFAULT_POLICY), help="Override ONNX policy path from config.")
    parser.add_argument("--policy_backend", choices=("torchscript", "onnx"), default="onnx")
    parser.add_argument("--model_path", default=None, help="Override pitch/roll MuJoCo XML path from config.")
    parser.add_argument("--real_ankle_model_path", default=None, help="XML with upper/lower ankle tendon geometry.")
    parser.add_argument("--cmd_vel", type=float, nargs=3, default=(0.0, 0.0, 0.0), help="Initial vx vy yaw_rate.")
    parser.add_argument("--sim_duration", type=float, default=30.0)
    parser.add_argument("--output_csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--render_mode", choices=("none", "window", "video"), default="none")
    parser.add_argument("--headless", action="store_true", help="Alias for --render_mode none.")
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--render_every", type=int, default=1, help="Render every N policy steps.")
    parser.add_argument("--sample_every", type=int, default=1, help="Write one row every N policy steps.")
    parser.add_argument("--video_path", type=Path, default=Path(__file__).resolve().parent / "data" / "policy_pr_ul.mp4")
    parser.add_argument("--viewer_width", type=int, default=1280)
    parser.add_argument("--viewer_height", type=int, default=720)
    parser.add_argument("--camera_distance", type=float, default=4.0)
    parser.add_argument("--camera_azimuth", type=float, default=45.0)
    parser.add_argument("--camera_elevation", type=float, default=-20.0)
    parser.add_argument("--init_height", type=float, default=None)
    parser.add_argument("--no_auto_base_height", action="store_true")
    parser.add_argument("--print_every", type=int, default=25)
    parser.add_argument("--jacobian_eps", type=float, default=1e-5)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
