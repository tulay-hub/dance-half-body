from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from legged_lab.envs import ManagerBasedAnimationEnv
    from legged_lab.managers import AnimationTerm


LEG_PITCH_ROLL_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]
LEG_PITCH_ROLL_JOINT_IDS = list(range(len(LEG_PITCH_ROLL_JOINT_NAMES)))
LEG_PITCH_ROLL_ASSET_CFG = SceneEntityCfg(
    "robot",
    joint_names=LEG_PITCH_ROLL_JOINT_NAMES,
    preserve_order=True,
)
LEG_KEY_BODY_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]
LEG_KEY_BODY_IDS = [0, 1]
SCRIPTED_ARM_SWING_JOINT_NAMES = [
    "torso_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
]


class LegDrivenArmSwingAction(ActionTerm):
    """Drive torso and arm joints from leg swing without adding policy actions."""

    cfg: LegDrivenArmSwingActionCfg

    def __init__(self, cfg: LegDrivenArmSwingActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names,
            preserve_order=self.cfg.preserve_order,
        )
        self._leg_joint_ids, self._leg_joint_names = self._asset.find_joints(
            LEG_PITCH_ROLL_JOINT_NAMES,
            preserve_order=True,
        )
        self._foot_body_ids, self._foot_body_names = self._asset.find_bodies(
            LEG_KEY_BODY_NAMES,
            preserve_order=True,
        )
        self._local_id = {name: i for i, name in enumerate(self._joint_names)}
        self._raw_actions = torch.zeros(self.num_envs, 0, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._export_IO_descriptor = False
        self._target_pos = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        return

    def apply_actions(self):
        self._update_arm_targets()
        self._asset.set_joint_position_target(
            self._target_pos,
            joint_ids=self._joint_ids,
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None or isinstance(env_ids, slice):
            self._target_pos[:] = self._asset.data.default_joint_pos[:, self._joint_ids]
        else:
            self._target_pos[env_ids] = self._asset.data.default_joint_pos[env_ids][:, self._joint_ids]

    def _update_arm_targets(self):
        default_pos = self._asset.data.default_joint_pos[:, self._joint_ids]
        foot_pos_w = self._asset.data.body_pos_w[:, self._foot_body_ids]
        root_pos_w = self._asset.data.root_pos_w
        root_quat_w = self._asset.data.root_quat_w
        foot_pos_b = math_utils.quat_apply_inverse(
            root_quat_w.unsqueeze(1).expand(-1, 2, -1),
            foot_pos_w - root_pos_w.unsqueeze(1),
        )
        foot_center_x = 0.5 * (foot_pos_b[:, 0, 0] + foot_pos_b[:, 1, 0])
        left_foot_swing = foot_pos_b[:, 0, 0] - foot_center_x
        right_foot_swing = foot_pos_b[:, 1, 0] - foot_center_x

        target = default_pos.clone()
        left_arm_pitch = torch.clamp(
            self.cfg.shoulder_pitch_gain * right_foot_swing,
            -self.cfg.shoulder_pitch_limit,
            self.cfg.shoulder_pitch_limit,
        )
        right_arm_pitch = torch.clamp(
            self.cfg.shoulder_pitch_gain * left_foot_swing,
            -self.cfg.shoulder_pitch_limit,
            self.cfg.shoulder_pitch_limit,
        )
        left_elbow = torch.clamp(
            self.cfg.elbow_gain * right_foot_swing.abs(),
            0.0,
            self.cfg.elbow_limit,
        )
        right_elbow = torch.clamp(
            self.cfg.elbow_gain * left_foot_swing.abs(),
            0.0,
            self.cfg.elbow_limit,
        )

        target[:, self._local_id["left_shoulder_pitch_joint"]] += left_arm_pitch
        target[:, self._local_id["right_shoulder_pitch_joint"]] += right_arm_pitch
        target[:, self._local_id["left_elbow_joint"]] += left_elbow
        target[:, self._local_id["right_elbow_joint"]] += right_elbow

        limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]
        target = torch.clamp(target, limits[..., 0], limits[..., 1])
        self._target_pos = self.cfg.smoothing * target + (1.0 - self.cfg.smoothing) * self._target_pos


@configclass
class LegDrivenArmSwingActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = LegDrivenArmSwingAction
    joint_names: list[str] = SCRIPTED_ARM_SWING_JOINT_NAMES
    preserve_order: bool = True
    shoulder_pitch_gain: float = -2.2
    shoulder_pitch_limit: float = 0.55
    elbow_gain: float = 1.0
    elbow_limit: float = 0.25
    smoothing: float = 0.35


def leg_pitch_roll_joint_pos(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = LEG_PITCH_ROLL_ASSET_CFG,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, asset_cfg.joint_ids]


def leg_pitch_roll_joint_pos_rel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = LEG_PITCH_ROLL_ASSET_CFG,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]


def leg_pitch_roll_joint_vel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = LEG_PITCH_ROLL_ASSET_CFG,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, asset_cfg.joint_ids]


def leg_pitch_roll_joint_vel_rel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = LEG_PITCH_ROLL_ASSET_CFG,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, asset_cfg.joint_ids] - asset.data.default_joint_vel[:, asset_cfg.joint_ids]


def leg_key_body_pos_b(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=LEG_KEY_BODY_NAMES, preserve_order=True),
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    key_body_pos_w = robot.data.body_pos_w[:, asset_cfg.body_ids, :]
    root_pos_w = robot.data.root_pos_w
    root_quat = robot.data.root_quat_w
    num_key_bodies = key_body_pos_w.shape[1]
    num_envs = root_pos_w.shape[0]
    key_body_pos = math_utils.quat_apply_inverse(
        root_quat.unsqueeze(1).expand(-1, num_key_bodies, -1),
        key_body_pos_w - root_pos_w.unsqueeze(1).expand(-1, num_key_bodies, -1),
    )
    return key_body_pos.reshape(num_envs, -1)


def ref_leg_pitch_roll_joint_pos(
    env: ManagerBasedAnimationEnv,
    animation: str,
    flatten_steps_dim: bool = True,
    joint_ids: list[int] | tuple[int, ...] = LEG_PITCH_ROLL_JOINT_IDS,
) -> torch.Tensor:
    animation_term: AnimationTerm = env.animation_manager.get_term(animation)
    ref_dof_pos = animation_term.get_dof_pos()[:, :, joint_ids]
    if flatten_steps_dim:
        return ref_dof_pos.reshape(env.num_envs, -1)
    return ref_dof_pos


def ref_leg_pitch_roll_joint_vel(
    env: ManagerBasedAnimationEnv,
    animation: str,
    flatten_steps_dim: bool = True,
    joint_ids: list[int] | tuple[int, ...] = LEG_PITCH_ROLL_JOINT_IDS,
) -> torch.Tensor:
    animation_term: AnimationTerm = env.animation_manager.get_term(animation)
    ref_dof_vel = animation_term.get_dof_vel()[:, :, joint_ids]
    if flatten_steps_dim:
        return ref_dof_vel.reshape(env.num_envs, -1)
    return ref_dof_vel


def ref_leg_key_body_pos_b(
    env: ManagerBasedAnimationEnv,
    animation: str,
    flatten_steps_dim: bool = True,
    body_ids: list[int] | tuple[int, ...] = LEG_KEY_BODY_IDS,
) -> torch.Tensor:
    animation_term: AnimationTerm = env.animation_manager.get_term(animation)
    ref_key_body_pos = animation_term.get_key_body_pos_b()[:, :, body_ids]
    if flatten_steps_dim:
        return ref_key_body_pos.reshape(env.num_envs, -1)
    return ref_key_body_pos.reshape(ref_key_body_pos.shape[0], ref_key_body_pos.shape[1], -1)
