"""Replay retargeted Lens110 motions in MuJoCo and collect PR/UL ankle data."""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import sys
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")

import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
T_P_V_DIR = REPO_ROOT / "pitchRoll2UpperLower" / "t_p_v"
SIM2SIM_DIR = REPO_ROOT / "scripts" / "sim2sim_ul"
for path in (T_P_V_DIR, SIM2SIM_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import collect_policy_pr_ul_data as base_collect  # noqa: E402
import ul_full as sim2sim  # noqa: E402


ANKLE_PR_NAMES = (
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)


def apply_config_overrides(args: argparse.Namespace, cfg) -> None:
    if args.clip_control_targets_to_joint_limits is not None:
        cfg.clip_control_targets_to_joint_limits = args.clip_control_targets_to_joint_limits
    if args.ankle_motor_kp_scale is not None:
        cfg.ankle_motor_kp_scale = args.ankle_motor_kp_scale
    if args.ankle_motor_kd_scale is not None:
        cfg.ankle_motor_kd_scale = args.ankle_motor_kd_scale
    if args.ankle_motor_tau_scale is not None:
        cfg.ankle_motor_tau_scale = args.ankle_motor_tau_scale
    if args.tracking_kp_scale != 1.0:
        cfg.kp = cfg.kp * args.tracking_kp_scale
    if args.tracking_kd_scale != 1.0:
        cfg.kd = cfg.kd * args.tracking_kd_scale

    args.model_path = sim2sim.resolve_path(args.model_path) if args.model_path else cfg.model_path
    args.ankle_model_path = (
        sim2sim.resolve_path(args.ankle_model_path) if args.ankle_model_path else cfg.ankle_model_path
    )
    args.init_height = args.init_height if args.init_height is not None else (cfg.init_height or cfg.fallback_init_height)
    args.print_every = args.print_every if args.print_every is not None else cfg.print_every


def resolve_motion_paths(path_arg: str) -> list[Path]:
    paths: list[Path] = []
    for item in path_arg.split(","):
        item = item.strip()
        if not item:
            continue
        path = Path(item)
        if path.is_dir():
            paths.extend(sorted(path.glob("*.npz")))
            paths.extend(sorted(path.glob("*.pkl")))
        else:
            paths.append(path)
    if not paths:
        raise ValueError("No retarget motion files provided.")
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing retarget motion files: " + ", ".join(missing))
    return paths


class CompatUnpickler(pickle.Unpickler):
    """Load NumPy-2 pickles in environments that still expose numpy.core."""

    def find_class(self, module: str, name: str):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def load_motion(path: Path) -> dict:
    if path.suffix == ".npz":
        raw = np.load(path, allow_pickle=True)
        return {key: raw[key] for key in raw.keys()}
    if path.suffix == ".pkl":
        with path.open("rb") as file:
            motion = CompatUnpickler(file).load()
        if not isinstance(motion, dict):
            raise TypeError(f"Expected dict in {path}, got {type(motion).__name__}")
        return motion
    raise ValueError(f"Unsupported motion file extension: {path}")


def root_rot_wxyz(motion: dict, path: Path) -> np.ndarray:
    root_rot = np.asarray(motion["root_rot"], dtype=np.float64)
    fmt_value = motion.get("root_rot_format", "wxyz")
    fmt = str(np.asarray(fmt_value).reshape(-1)[0]).lower()
    if fmt == "xyzw":
        return root_rot[:, [3, 0, 1, 2]]
    if fmt == "wxyz":
        return root_rot
    raise ValueError(f"Unsupported root_rot_format={fmt!r} in {path}")


def finite_diff(values: np.ndarray, dt: float) -> np.ndarray:
    vel = np.zeros_like(values)
    if len(values) <= 1:
        return vel
    vel[1:-1] = (values[2:] - values[:-2]) / (2.0 * dt)
    vel[0] = (values[1] - values[0]) / dt
    vel[-1] = (values[-1] - values[-2]) / dt
    return vel


def foot_min_z(model: mujoco.MjModel, data: mujoco.MjData, cfg) -> float | None:
    foot_geom_ids = sim2sim.collect_foot_geom_ids(model, cfg.foot_contact_geoms)
    if not foot_geom_ids:
        return None
    return min(sim2sim.geom_min_z(model, data, geom_id) for geom_id in foot_geom_ids)


def pitch_roll_to_upper_lower_jacobian(
    ankle_mapper,
    side: str,
    pitch: float,
    roll: float,
    eps: float = 1e-5,
) -> np.ndarray:
    if hasattr(ankle_mapper, "jacobian"):
        return ankle_mapper.jacobian(side, pitch, roll, eps)

    observe = ankle_mapper.robot.default_pos.copy()
    observe_index = ankle_mapper.observe_index
    control_index = ankle_mapper.control_index
    if side == "left":
        pitch_name, roll_name = "left_ankle_pitch_joint", "left_ankle_roll_joint"
        upper_name, lower_name = "left_ankle_upper_joint", "left_ankle_lower_joint"
    elif side == "right":
        pitch_name, roll_name = "right_ankle_pitch_joint", "right_ankle_roll_joint"
        upper_name, lower_name = "right_ankle_upper_joint", "right_ankle_lower_joint"
    else:
        raise ValueError(side)

    def eval_ul(pitch_value: float, roll_value: float) -> np.ndarray:
        observe[observe_index[pitch_name]] = pitch_value
        observe[observe_index[roll_name]] = roll_value
        control = ankle_mapper.observe_to_control_target(observe)
        return control[[control_index[upper_name], control_index[lower_name]]]

    upper_pitch = eval_ul(pitch + eps, roll)
    lower_pitch = eval_ul(pitch - eps, roll)
    upper_roll = eval_ul(pitch, roll + eps)
    lower_roll = eval_ul(pitch, roll - eps)
    return np.column_stack(((upper_pitch - lower_pitch) / (2.0 * eps), (upper_roll - lower_roll) / (2.0 * eps)))


def add_equivalent_pitch_roll_torque(
    row: dict[str, float | int | str],
    ankle_mapper,
    data: mujoco.MjData,
    robot,
    ul_ids: np.ndarray,
) -> None:
    if len(ul_ids) != 4:
        return
    tau_ul = data.actuator_force[robot.actuator_ids[ul_ids]]
    left_jac = pitch_roll_to_upper_lower_jacobian(
        ankle_mapper,
        "left",
        float(row["left_pitch_pos"]),
        float(row["left_roll_pos"]),
    )
    right_jac = pitch_roll_to_upper_lower_jacobian(
        ankle_mapper,
        "right",
        float(row["right_pitch_pos"]),
        float(row["right_roll_pos"]),
    )
    left_tau_pr = left_jac.T @ tau_ul[0:2]
    right_tau_pr = right_jac.T @ tau_ul[2:4]
    row["left_pitch_tau"] = float(left_tau_pr[0])
    row["left_roll_tau"] = float(left_tau_pr[1])
    row["right_pitch_tau"] = float(right_tau_pr[0])
    row["right_roll_tau"] = float(right_tau_pr[1])


def align_dof_pos(motion: dict, cfg) -> tuple[np.ndarray, np.ndarray]:
    dof_names = [str(name) for name in np.asarray(motion["dof_names"]).tolist()]
    dof_pos_raw = np.asarray(motion["dof_pos"], dtype=np.float64)
    if "dof_vel" in motion:
        dof_vel_raw = np.asarray(motion["dof_vel"], dtype=np.float64)
    else:
        fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
        dof_vel_raw = finite_diff(dof_pos_raw, 1.0 / fps)
    source_index = {name: i for i, name in enumerate(dof_names)}
    dof_pos = np.tile(np.asarray(cfg.default_pos, dtype=np.float64), (dof_pos_raw.shape[0], 1))
    dof_vel = np.zeros_like(dof_pos)
    for target_i, name in enumerate(cfg.mujoco_joint_names):
        if name in source_index:
            dof_pos[:, target_i] = dof_pos_raw[:, source_index[name]]
            dof_vel[:, target_i] = dof_vel_raw[:, source_index[name]]
    return dof_pos, dof_vel


def set_motion_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot,
    cfg,
    root_pos: np.ndarray,
    root_rot_wxyz: np.ndarray,
    dof_pos: np.ndarray,
    dof_vel: np.ndarray,
    control_target_pos: np.ndarray,
    control_target_vel: np.ndarray,
) -> None:
    data.qpos[:3] = root_pos
    data.qpos[3:7] = root_rot_wxyz
    data.qpos[robot.qpos_ids] = dof_pos
    data.qpos[robot.control_qpos_ids] = control_target_pos
    data.qvel[:] = 0.0
    data.qvel[robot.qvel_ids] = dof_vel
    data.qvel[robot.control_qvel_ids] = control_target_vel
    data.ctrl[:] = 0.0
    data.ctrl[robot.actuator_ids] = control_target_pos
    mujoco.mj_forward(model, data)


def initialize_motion_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot,
    root_pos: np.ndarray,
    root_rot_wxyz: np.ndarray,
    dof_pos: np.ndarray,
    control_target_pos: np.ndarray,
) -> None:
    data.qpos[:3] = root_pos
    data.qpos[3:7] = root_rot_wxyz
    data.qpos[robot.qpos_ids] = dof_pos
    data.qpos[robot.control_qpos_ids] = control_target_pos
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    data.ctrl[robot.actuator_ids] = control_target_pos
    mujoco.mj_forward(model, data)


def upright_root_rotation(num_frames: int) -> np.ndarray:
    root_rot = np.zeros((num_frames, 4), dtype=np.float64)
    root_rot[:, 0] = 1.0
    return root_rot


def quat_to_rotmat_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    quat = quat / max(float(np.linalg.norm(quat)), 1.0e-12)
    w, x, y, z = quat
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def apply_base_assist(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    args: argparse.Namespace,
    pelvis_body_id: int,
    target_xy: np.ndarray,
    target_z: float,
) -> None:
    data.xfrc_applied[:, :] = 0.0
    if args.no_base_assist:
        return

    force = np.zeros(3, dtype=np.float64)
    force[:2] = args.base_assist_xy_kp * (target_xy - data.qpos[:2]) - args.base_assist_xy_kd * data.qvel[:2]
    force[2] = args.base_assist_z_kp * (target_z - data.qpos[2]) - args.base_assist_z_kd * data.qvel[2]
    robot_mass = float(np.sum(model.body_mass[1:]))
    force[2] += args.base_assist_gravity_scale * robot_mass * abs(float(model.opt.gravity[2]))
    force = np.clip(force, -args.base_assist_max_force, args.base_assist_max_force)

    rot = quat_to_rotmat_wxyz(data.qpos[3:7])
    body_z_world = rot[:, 2]
    tilt_axis_world = np.cross(body_z_world, np.array([0.0, 0.0, 1.0], dtype=np.float64))
    torque = args.base_assist_tilt_kp * tilt_axis_world - args.base_assist_tilt_kd * data.qvel[3:6]
    torque = np.clip(torque, -args.base_assist_max_torque, args.base_assist_max_torque)

    data.xfrc_applied[pelvis_body_id, :3] = force
    data.xfrc_applied[pelvis_body_id, 3:] = torque


def unstable_state(data: mujoco.MjData, args: argparse.Namespace) -> bool:
    if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
        return True
    if data.qpos[2] < args.min_root_z:
        return True
    if abs(float(data.qvel[3])) > args.max_abs_roll_rate:
        return True
    if abs(float(data.qvel[4])) > args.max_abs_pitch_rate:
        return True
    return False


def replay_one_motion(
    motion_path: Path,
    args: argparse.Namespace,
    cfg,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot,
    ankle_mapper,
    ankle_pr_ids: np.ndarray,
    ul_ids: np.ndarray,
    pelvis_body_id: int,
    writer_state: dict,
    csv_file,
    renderer,
    camera,
    video_writer,
    viewer,
) -> int:
    motion = load_motion(motion_path)
    fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
    source_root_pos = np.asarray(motion["root_pos"], dtype=np.float64)
    if args.use_motion_root_rotation:
        source_root_rot = root_rot_wxyz(motion, motion_path)
    else:
        source_root_rot = upright_root_rotation(len(source_root_pos))
    dof_pos_all, dof_vel_all = align_dof_pos(motion, cfg)
    frame_dt = 1.0 / fps
    control_target_all = []
    for target_pos in dof_pos_all:
        clipped_target = target_pos.copy()
        sim2sim.clip_policy_target(clipped_target, robot, cfg)
        control_target = ankle_mapper.observe_to_control_target(clipped_target)
        sim2sim.clip_control_target(control_target, robot, cfg)
        control_target_all.append(control_target)
    control_target_all = np.asarray(control_target_all, dtype=np.float64)
    control_target_vel_all = finite_diff(control_target_all, frame_dt)
    total_frames = len(dof_pos_all)
    max_frames = total_frames if args.max_frames is None else min(total_frames, args.max_frames)
    frame_step = max(1, int(args.frame_step))
    rows = 0
    start_time = time.time()
    print(f"[REPLAY] {motion_path.name}: frames={max_frames}/{total_frames} fps={fps:.1f}")

    if not args.pin_root:
        init_target = dof_pos_all[0].copy()
        sim2sim.clip_policy_target(init_target, robot, cfg)
        init_control_target = control_target_all[0].copy()
        init_root_pos = source_root_pos[0].copy()
        if args.keep_root_xy_fixed:
            init_root_pos[:2] = data.qpos[:2]
        if args.keep_root_z_from_stand:
            init_root_pos[2] = data.qpos[2]
        initialize_motion_state(
            model,
            data,
            robot,
            init_root_pos,
            source_root_rot[0].copy(),
            init_target,
            init_control_target,
        )
        base_assist_target_xy = init_root_pos[:2].copy()
        base_assist_target_z = float(init_root_pos[2] if args.base_assist_height is None else args.base_assist_height)
    else:
        base_assist_target_xy = source_root_pos[0, :2].copy()
        base_assist_target_z = float(source_root_pos[0, 2] if args.base_assist_height is None else args.base_assist_height)

    prev_target_pos = dof_pos_all[0].copy()
    prev_ul_pos = data.qpos[robot.control_qpos_ids][ul_ids].copy()
    prev_ul_target = control_target_all[0][ul_ids].copy()

    for frame in range(0, max_frames, frame_step):
        root_pos = source_root_pos[frame].copy()
        root_rot = source_root_rot[frame].copy()
        target_pos = dof_pos_all[frame].copy()
        target_vel = dof_vel_all[frame].copy()
        sim2sim.clip_policy_target(target_pos, robot, cfg)
        control_target_pos = control_target_all[frame].copy()
        control_target_vel = control_target_vel_all[frame].copy()

        if args.pin_root:
            set_motion_state(
                model,
                data,
                robot,
                cfg,
                root_pos,
                root_rot,
                target_pos,
                target_vel,
                control_target_pos,
                control_target_vel,
            )
        else:
            data.ctrl[robot.actuator_ids] = control_target_pos
            steps_per_frame = max(1, int(round(frame_dt / cfg.dt)))
            for _ in range(steps_per_frame):
                apply_base_assist(
                    model,
                    data,
                    args,
                    pelvis_body_id,
                    base_assist_target_xy,
                    base_assist_target_z,
                )
                mujoco.mj_step(model, data)

        qpos, qvel, lin_vel_b, omega_b, _ = sim2sim.get_base_obs(data)
        q = qpos[robot.qpos_ids]
        dq = qvel[robot.qvel_ids]
        tau = data.actuator_force[robot.actuator_ids].copy()
        ul_pos = data.qpos[robot.control_qpos_ids][ul_ids].copy()
        ul_target = control_target_pos[ul_ids].copy()
        left_jac = pitch_roll_to_upper_lower_jacobian(
            ankle_mapper,
            "left",
            float(q[ankle_pr_ids[0]]),
            float(q[ankle_pr_ids[1]]),
        )
        right_jac = pitch_roll_to_upper_lower_jacobian(
            ankle_mapper,
            "right",
            float(q[ankle_pr_ids[2]]),
            float(q[ankle_pr_ids[3]]),
        )
        row = base_collect.row_from_state(
            frame,
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
            tau[ul_ids],
            0.0,
            0.0,
            float(np.linalg.cond(left_jac)),
            float(np.linalg.cond(right_jac)),
        )
        add_equivalent_pitch_roll_torque(row, ankle_mapper, data, robot, ul_ids)
        min_z = foot_min_z(model, data, cfg)
        if min_z is not None:
            row["foot_min_z"] = float(min_z)
        row["motion_name"] = motion_path.stem
        row["motion_frame"] = int(frame)
        row["motion_time"] = float(frame * frame_dt)
        row["pin_root"] = int(args.pin_root)
        row["use_motion_root_rotation"] = int(args.use_motion_root_rotation)
        row["base_assist"] = int(not args.no_base_assist)
        row["base_assist_target_z"] = float(base_assist_target_z)
        row["base_assist_gravity_scale"] = float(args.base_assist_gravity_scale)
        row["root_z"] = float(qpos[2])

        prev_target_pos = target_pos.copy()
        prev_ul_pos = ul_pos.copy()
        prev_ul_target = ul_target.copy()

        if writer_state.get("writer") is None:
            writer_state["writer"] = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
            writer_state["writer"].writeheader()
        writer_state["writer"].writerow(row)
        rows += 1

        if args.print_every and rows % args.print_every == 0:
            print(f"[REPLAY] {motion_path.name} frame={frame}/{max_frames} rows={rows}")
        if args.stop_on_unstable and not args.pin_root and unstable_state(data, args):
            print(
                "[INFO] stability guard triggered; stopping motion. "
                f"root_z={qpos[2]:.3f}, qvel_ang=({qvel[3]:+.2f},{qvel[4]:+.2f},{qvel[5]:+.2f})"
            )
            break
        if rows % args.render_every == 0:
            if not base_collect.render(data, renderer, camera, video_writer, viewer, args.render_mode):
                print("[INFO] viewer closed; stopping replay")
                break
        if args.realtime:
            elapsed = time.time() - start_time
            target_elapsed = rows * frame_step * frame_dt
            if elapsed < target_elapsed:
                time.sleep(target_elapsed - elapsed)
    return rows


def build_runtime(args: argparse.Namespace, cfg):
    model = mujoco.MjModel.from_xml_path(args.model_path)
    model.opt.timestep = cfg.dt
    data = mujoco.MjData(model)
    robot = sim2sim.make_robot(SimpleNamespace(init_height=args.init_height), cfg)
    if cfg.use_training_passive_ankle_limits:
        sim2sim.patch_training_passive_ankle_limits(model)
    (
        robot.qpos_ids,
        robot.qvel_ids,
        robot.control_qpos_ids,
        robot.control_qvel_ids,
        robot.actuator_ids,
        robot.observe_target_min,
        robot.observe_target_max,
        robot.control_target_min,
        robot.control_target_max,
    ) = sim2sim.build_joint_data(model, cfg)
    ankle_mapper = sim2sim.PolynomialAnkleMapper(args.ankle_model_path, robot, cfg)
    robot.control_default_pos = ankle_mapper.observe_to_control_target(robot.default_pos)
    sim2sim.patch_model(model, robot, cfg, torque_control_ids=set())
    if not args.no_auto_base_height:
        robot.base_pos[2] = sim2sim.grounded_base_height(model, data, robot, cfg)
    sim2sim.hold_default_stand(data, robot, cfg)
    mujoco.mj_forward(model, data)
    mujoco_index = {name: i for i, name in enumerate(cfg.mujoco_joint_names)}
    robot.name_to_mujoco = mujoco_index
    ankle_pr_ids = np.array([mujoco_index[name] for name in ANKLE_PR_NAMES], dtype=np.int32)
    ul_ids = sim2sim.upper_lower_control_ids(cfg)
    pelvis_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if pelvis_body_id < 0:
        raise ValueError("Could not find MuJoCo body: pelvis")
    return model, data, robot, ankle_mapper, ankle_pr_ids, ul_ids, pelvis_body_id


def run(args: argparse.Namespace, cfg) -> None:
    motion_paths = resolve_motion_paths(args.motion)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output_path}. Use --overwrite or choose another path.")

    model, data, robot, ankle_mapper, ankle_pr_ids, ul_ids, pelvis_body_id = build_runtime(args, cfg)
    renderer, camera, video_writer, viewer = base_collect.init_render(model, data, args, cfg)
    total_rows = 0
    writer_state: dict = {"writer": None}
    with output_path.open("w", newline="") as csv_file:
        try:
            for motion_path in motion_paths:
                total_rows += replay_one_motion(
                    motion_path,
                    args,
                    cfg,
                    model,
                    data,
                    robot,
                    ankle_mapper,
                    ankle_pr_ids,
                    ul_ids,
                    pelvis_body_id,
                    writer_state,
                    csv_file,
                    renderer,
                    camera,
                    video_writer,
                    viewer,
                )
        finally:
            if video_writer is not None:
                video_writer.release()
                print(f"[INFO] saved video: {args.video_path}")
            if viewer is not None:
                viewer.close()
    print(f"[INFO] wrote {total_rows} rows: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay retargeted Lens110 npz/pkl motions and collect PR/UL ankle data.")
    parser.add_argument(
        "--motion",
        default=str(REPO_ROOT / "source/legged_lab/legged_lab/data/MotionData/lens110_gmr_lab"),
        help="Local Lens110 retargeted motion file or directory.",
    )
    parser.add_argument("--output_csv", default=str(T_P_V_DIR / "data" / "retarget_pkl_pr_ul.csv"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--config", default=str(SIM2SIM_DIR / "ul.json"))
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--ankle_model_path", default=None)
    parser.add_argument("--init_height", type=float, default=None)
    parser.add_argument("--no_auto_base_height", action="store_true")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--frame_step", type=int, default=1)
    parser.add_argument("--pin_root", action="store_true", default=False, help="Debug mode: directly set qpos/root every frame.")
    parser.add_argument("--dynamic_root", dest="pin_root", action="store_false")
    parser.add_argument("--keep_root_xy_fixed", action="store_true", default=True)
    parser.add_argument("--move_root_xy_from_motion", dest="keep_root_xy_fixed", action="store_false")
    parser.add_argument("--keep_root_z_from_stand", action="store_true", default=True)
    parser.add_argument("--use_motion_root_z", dest="keep_root_z_from_stand", action="store_false")
    parser.add_argument(
        "--use_motion_root_rotation",
        action="store_true",
        help="Use root_rot from the motion file. Default keeps the floating base upright.",
    )
    parser.add_argument("--clip_control_targets_to_joint_limits", dest="clip_control_targets_to_joint_limits", action="store_true", default=None)
    parser.add_argument("--no_clip_control_targets_to_joint_limits", dest="clip_control_targets_to_joint_limits", action="store_false")
    parser.add_argument("--ankle_motor_kp_scale", type=float, default=None)
    parser.add_argument("--ankle_motor_kd_scale", type=float, default=None)
    parser.add_argument("--ankle_motor_tau_scale", type=float, default=None)
    parser.add_argument("--tracking_kp_scale", type=float, default=1.0)
    parser.add_argument("--tracking_kd_scale", type=float, default=1.0)
    parser.add_argument("--no_base_assist", action="store_true")
    parser.add_argument("--base_assist_height", type=float, default=None)
    parser.add_argument("--base_assist_gravity_scale", type=float, default=0.5)
    parser.add_argument("--base_assist_z_kp", type=float, default=120.0)
    parser.add_argument("--base_assist_z_kd", type=float, default=45.0)
    parser.add_argument("--base_assist_xy_kp", type=float, default=0.0)
    parser.add_argument("--base_assist_xy_kd", type=float, default=8.0)
    parser.add_argument("--base_assist_tilt_kp", type=float, default=45.0)
    parser.add_argument("--base_assist_tilt_kd", type=float, default=16.0)
    parser.add_argument("--base_assist_max_force", type=float, default=260.0)
    parser.add_argument("--base_assist_max_torque", type=float, default=35.0)
    parser.add_argument("--stop_on_unstable", action="store_true", default=True)
    parser.add_argument("--no_stop_on_unstable", dest="stop_on_unstable", action="store_false")
    parser.add_argument("--min_root_z", type=float, default=0.30)
    parser.add_argument("--max_abs_roll_rate", type=float, default=8.0)
    parser.add_argument("--max_abs_pitch_rate", type=float, default=8.0)
    parser.add_argument("--render_mode", choices=("none", "window", "video"), default="none")
    parser.add_argument("--render_every", type=int, default=1)
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--video_path", default=str(T_P_V_DIR / "data" / "retarget_pkl_pr_ul.mp4"))
    parser.add_argument("--viewer_width", type=int, default=1280)
    parser.add_argument("--viewer_height", type=int, default=720)
    parser.add_argument("--print_every", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = sim2sim.load_config(args.config)
    apply_config_overrides(args, cfg)
    run(args, cfg)


if __name__ == "__main__":
    main()
