"""View and compare two Lens110 initial poses in MuJoCo.

Keys in the MuJoCo viewer:
    1 / 2 / 3: select pose
    Space: next pose

Run:
    python -u pitchRoll2UpperLower/t_p_v/view_lens110_initial_poses.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _wants_headless() -> bool:
    return "--headless" in sys.argv or "--render_mode=none" in sys.argv


if _wants_headless():
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")

import mujoco

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_MODEL = (
    REPO_ROOT
    / "source"
    / "legged_lab"
    / "legged_lab"
    / "data"
    / "Robots"
    / "model_humanoid_lens110"
    / "mjcf"
    / "lens110.xml"
)
UL_OUTPUT_TO_JOINT = {
    "left_upper_pos": "left_ankle_upper_joint",
    "left_lower_pos": "left_ankle_lower_joint",
    "right_upper_pos": "right_ankle_upper_joint",
    "right_lower_pos": "right_ankle_lower_joint",
}


MJCF_UL_DEFAULT_POSE = {
    "left_hip_pitch_joint": -0.14,
    "left_hip_roll_joint": 0.01,
    "left_hip_yaw_joint": -0.1,
    "left_knee_joint": 0.30,
    "left_ankle_pitch_joint": -0.15,
    "left_ankle_roll_joint": 0.0,
    "left_ankle_upper_joint": 0.161341,
    "left_ankle_lower_joint": 0.162273,
    "right_hip_pitch_joint": -0.14,
    "right_hip_roll_joint": -0.01,
    "right_hip_yaw_joint": 0.1,
    "right_knee_joint": 0.30,
    "right_ankle_pitch_joint": -0.15,
    "right_ankle_roll_joint": 0.0,
    "right_ankle_upper_joint": 0.161341,
    "right_ankle_lower_joint": 0.162273,
    "torso_yaw_joint": 0.0,
    "left_shoulder_pitch_joint": 0.4,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": -0.8,
    "right_shoulder_pitch_joint": 0.4,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.8,
}


URDF_PR_DEFAULT_POSE_IN_MJCF = {
    "left_hip_pitch_joint": -0.14,
    "left_hip_roll_joint": 0.01,
    "left_hip_yaw_joint": -0.1,
    "left_knee_joint": 0.30,
    "left_ankle_pitch_joint": -0.15,
    "left_ankle_roll_joint": -0.0,
    "right_hip_pitch_joint": -0.14,
    "right_hip_roll_joint": -0.01,
    "right_hip_yaw_joint": 0.1,
    "right_knee_joint": 0.30,
    "right_ankle_pitch_joint": -0.15,
    "right_ankle_roll_joint": -0.0,
    "torso_yaw_joint": 0.0,
    "left_shoulder_pitch_joint": 0.4,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": -0.8,
    "right_shoulder_pitch_joint": 0.4,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_yaw_joint": 0.0,
    # The URDF/21DoF model uses right_elbow axis="0 1 0", while lens110.xml
    # uses axis="0 -1 0". The equivalent MJCF visual pose is therefore +0.8.
    "right_elbow_joint": 0.8,
}


def joint_qpos_addr(model: mujoco.MjModel, joint_name: str) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise KeyError(f"Joint not found in MuJoCo model: {joint_name}")
    return int(model.jnt_qposadr[joint_id])


def reset_root(model: mujoco.MjModel, data: mujoco.MjData, root_z: float) -> None:
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            adr = int(model.jnt_qposadr[joint_id])
            data.qpos[adr : adr + 3] = [0.0, 0.0, root_z]
            data.qpos[adr + 3 : adr + 7] = [1.0, 0.0, 0.0, 0.0]
            data.qvel[int(model.jnt_dofadr[joint_id]) : int(model.jnt_dofadr[joint_id]) + 6] = 0.0
            return


def apply_pose(model: mujoco.MjModel, data: mujoco.MjData, pose: dict[str, float], root_z: float) -> None:
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    reset_root(model, data, root_z)
    for name, value in pose.items():
        data.qpos[joint_qpos_addr(model, name)] = float(value)
    mujoco.mj_forward(model, data)


def converted_ul_pose(base_pose: dict[str, float], backend: str) -> dict[str, float]:
    sys.path.insert(0, str(SCRIPT_DIR))
    from ankle_conversion_backends import ConversionBackends

    converter = ConversionBackends()
    pr_state = [
        base_pose["left_ankle_pitch_joint"],
        base_pose["left_ankle_roll_joint"],
        base_pose["right_ankle_pitch_joint"],
        base_pose["right_ankle_roll_joint"],
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    names, values = converter.convert(backend, "pr_to_ul", pr_state)
    pose = dict(base_pose)
    for name, value in zip(names[:4], values[:4]):
        pose[UL_OUTPUT_TO_JOINT[name]] = float(value)
    return pose


def make_pose_list(include_converted: bool, backend: str) -> list[tuple[str, dict[str, float]]]:
    poses = [
        ("1: MJCF default, PR + explicit UL, right_elbow=+0.8", MJCF_UL_DEFAULT_POSE),
        ("2: URDF/Isaac PR default mapped to lens110.xml, UL left at 0, right_elbow=+0.8", URDF_PR_DEFAULT_POSE_IN_MJCF),
    ]
    if include_converted:
        poses.append((f"3: URDF/Isaac PR default + {backend} converted UL", converted_ul_pose(URDF_PR_DEFAULT_POSE_IN_MJCF, backend)))
    return poses


def make_custom_pose_list(args: argparse.Namespace) -> list[tuple[str, dict[str, float]]] | None:
    if not args.custom_pose_json:
        return None
    pose = dict(MJCF_UL_DEFAULT_POSE)
    pose.update({str(name): float(value) for name, value in json.loads(args.custom_pose_json).items()})
    return [(args.custom_pose_label, pose)]


def print_pose_values(label: str, pose: dict[str, float]) -> None:
    keys = [
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "left_ankle_upper_joint",
        "left_ankle_lower_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "right_ankle_upper_joint",
        "right_ankle_lower_joint",
        "left_elbow_joint",
        "right_elbow_joint",
    ]
    print(f"\n[POSE] {label}")
    for key in keys:
        print(f"  {key:28s} {pose.get(key, 0.0): .6f}")


def run_window(args: argparse.Namespace) -> None:
    import mujoco.viewer

    model = mujoco.MjModel.from_xml_path(str(args.model_path))
    data = mujoco.MjData(model)
    poses = make_custom_pose_list(args) or make_pose_list(args.include_converted_ul, args.convert_backend)
    selected = {"index": max(0, min(args.pose - 1, len(poses) - 1))}
    last_switch = {"time": time.monotonic()}

    def select(index: int) -> None:
        selected["index"] = index % len(poses)
        label, pose = poses[selected["index"]]
        apply_pose(model, data, pose, args.root_z)
        print_pose_values(label, pose)

    def key_callback(keycode: int) -> None:
        if keycode in (ord("1"), ord("2"), ord("3")):
            index = keycode - ord("1")
            if index < len(poses):
                select(index)
        elif keycode == ord(" "):
            select(selected["index"] + 1)

    select(selected["index"])
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.cam.distance = args.camera_distance
        viewer.cam.azimuth = args.camera_azimuth
        viewer.cam.elevation = args.camera_elevation
        viewer.cam.lookat[:] = [0.0, 0.0, args.root_z * 0.65]
        while viewer.is_running():
            if args.auto_switch > 0.0 and time.monotonic() - last_switch["time"] >= args.auto_switch:
                select(selected["index"] + 1)
                last_switch["time"] = time.monotonic()
            label, pose = poses[selected["index"]]
            apply_pose(model, data, pose, args.root_z)
            viewer.sync()
            time.sleep(1.0 / args.fps)


def run_headless(args: argparse.Namespace) -> None:
    model = mujoco.MjModel.from_xml_path(str(args.model_path))
    data = mujoco.MjData(model)
    poses = make_custom_pose_list(args) or make_pose_list(args.include_converted_ul, args.convert_backend)
    for label, pose in poses:
        apply_pose(model, data, pose, args.root_z)
        print_pose_values(label, pose)
        print(f"  root_z                       {data.qpos[2]: .6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="View two Lens110 initial poses in MuJoCo.")
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--root_z", type=float, default=0.68)
    parser.add_argument("--pose", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--include_converted_ul", action="store_true", default=True)
    parser.add_argument("--no_converted_ul", action="store_false", dest="include_converted_ul")
    parser.add_argument("--convert_backend", choices=["weighted_mlp", "poly_degree3", "old_pkl"], default="old_pkl")
    parser.add_argument("--render_mode", choices=["window", "none"], default="window")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--auto_switch", type=float, default=0.0, help="Seconds between automatic pose switches; 0 disables it.")
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--camera_distance", type=float, default=2.0)
    parser.add_argument("--camera_azimuth", type=float, default=150.0)
    parser.add_argument("--camera_elevation", type=float, default=-15.0)
    parser.add_argument("--custom_pose_json", default=None, help="JSON dict of joint_name -> value to view as one custom pose.")
    parser.add_argument("--custom_pose_label", default="Custom pose")
    args = parser.parse_args()

    if args.headless or args.render_mode == "none":
        run_headless(args)
    else:
        run_window(args)


if __name__ == "__main__":
    main()
