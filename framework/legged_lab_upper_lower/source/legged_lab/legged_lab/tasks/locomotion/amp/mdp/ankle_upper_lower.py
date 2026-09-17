from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

import isaaclab.utils.string as string_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from legged_lab.envs import ManagerBasedAnimationEnv


PITCH_ROLL_JOINT_NAMES = (
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)
UPPER_LOWER_JOINT_NAMES = (
    "left_ankle_upper_joint",
    "left_ankle_lower_joint",
    "right_ankle_upper_joint",
    "right_ankle_lower_joint",
)

LENS110_MOTION_JOINT_NAMES = (
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
    "torso_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
)

LENS110_UPPER_LOWER_POLICY_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_upper_joint",
    "left_ankle_lower_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_upper_joint",
    "right_ankle_lower_joint",
    "torso_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
)


def _default_ankle_model_path() -> str:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "pitchRoll2UpperLower" / "ankle_model" / "models" / "ankle_poly_models.npz"
        if candidate.exists():
            return str(candidate)
    return str(
        Path(__file__).resolve().parents[7]
        / "pitchRoll2UpperLower"
        / "ankle_model"
        / "models"
        / "ankle_poly_models.npz"
    )


DEFAULT_ANKLE_MODEL_PATH = _default_ankle_model_path()

_MODEL_CACHE: dict[tuple[str, str, torch.dtype], "AnklePolynomialConverter"] = {}


class _PolynomialRidgeModel:
    def __init__(self, powers: np.ndarray, coef: np.ndarray, intercept: np.ndarray, device: torch.device, dtype: torch.dtype):
        self.powers = torch.as_tensor(powers, device=device, dtype=torch.int64)
        self.coef = torch.as_tensor(coef, device=device, dtype=dtype)
        self.intercept = torch.as_tensor(intercept, device=device, dtype=dtype)

    def __call__(self, values: torch.Tensor) -> torch.Tensor:
        values_shape = values.shape
        values = values.reshape(-1, 4)
        features = torch.prod(torch.pow(values.unsqueeze(1), self.powers.unsqueeze(0)), dim=-1)
        output = features @ self.coef.transpose(0, 1) + self.intercept
        return output.reshape(*values_shape[:-1], 4)


class AnklePolynomialConverter:
    """Torch evaluator for sklearn PolynomialFeatures + Ridge ankle mapping models."""

    def __init__(self, model_path: str, device: torch.device, dtype: torch.dtype):
        model_data = np.load(model_path)
        self._pos_pr2ul = _PolynomialRidgeModel(
            model_data["pos_pr2ul_powers"],
            model_data["pos_pr2ul_coef"],
            model_data["pos_pr2ul_intercept"],
            device,
            dtype,
        )
        self._vel_pr2ul = _PolynomialRidgeModel(
            model_data["vel_pr2ul_powers"],
            model_data["vel_pr2ul_coef"],
            model_data["vel_pr2ul_intercept"],
            device,
            dtype,
        )
        self._pos_ul2pr = _PolynomialRidgeModel(
            model_data["pos_ul2pr_powers"],
            model_data["pos_ul2pr_coef"],
            model_data["pos_ul2pr_intercept"],
            device,
            dtype,
        )
        self._vel_ul2pr = _PolynomialRidgeModel(
            model_data["vel_ul2pr_powers"],
            model_data["vel_ul2pr_coef"],
            model_data["vel_ul2pr_intercept"],
            device,
            dtype,
        )

    def _pr2ul_sidewise(self, values: torch.Tensor, model: _PolynomialRidgeModel) -> torch.Tensor:
        values_shape = values.shape
        values = values.reshape(-1, 4)
        left_inputs = torch.stack((values[:, 0], values[:, 1], values[:, 0], -values[:, 1]), dim=-1)
        right_inputs = torch.stack((values[:, 2], -values[:, 3], values[:, 2], values[:, 3]), dim=-1)
        left_outputs = model(left_inputs)
        right_outputs = model(right_inputs)
        outputs = torch.empty_like(values)
        outputs[:, 0:2] = left_outputs[:, 0:2]
        outputs[:, 2:4] = right_outputs[:, 2:4]
        return outputs.reshape(values_shape)

    def _ul2pr_sidewise(self, values: torch.Tensor, model: _PolynomialRidgeModel) -> torch.Tensor:
        values_shape = values.shape
        values = values.reshape(-1, 4)
        left_inputs = torch.stack((values[:, 0], values[:, 1], values[:, 0], values[:, 1]), dim=-1)
        right_inputs = torch.stack((values[:, 2], values[:, 3], values[:, 2], values[:, 3]), dim=-1)
        left_outputs = model(left_inputs)
        right_outputs = model(right_inputs)
        outputs = torch.empty_like(values)
        outputs[:, 0:2] = left_outputs[:, 0:2]
        outputs[:, 2:4] = right_outputs[:, 2:4]
        return outputs.reshape(values_shape)

    def pos_pr2ul(self, values: torch.Tensor) -> torch.Tensor:
        return self._pr2ul_sidewise(values, self._pos_pr2ul)

    def vel_pr2ul(self, values: torch.Tensor) -> torch.Tensor:
        return self._pr2ul_sidewise(values, self._vel_pr2ul)

    def pos_ul2pr(self, values: torch.Tensor) -> torch.Tensor:
        return self._ul2pr_sidewise(values, self._pos_ul2pr)

    def vel_ul2pr(self, values: torch.Tensor) -> torch.Tensor:
        return self._ul2pr_sidewise(values, self._vel_ul2pr)


def get_ankle_converter(
    model_path: str = DEFAULT_ANKLE_MODEL_PATH,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> AnklePolynomialConverter:
    device = torch.device(device)
    cache_key = (str(Path(model_path).resolve()), str(device), dtype)
    if cache_key not in _MODEL_CACHE:
        _MODEL_CACHE[cache_key] = AnklePolynomialConverter(cache_key[0], device, dtype)
    return _MODEL_CACHE[cache_key]


def joint_pos_upper_lower(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    model_path: str = DEFAULT_ANKLE_MODEL_PATH,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos, joint_names = _selected_joint_data(asset, asset.data.joint_pos, asset_cfg)
    return _replace_pitch_roll_with_upper_lower(joint_pos, joint_names, model_path, "pos")


def joint_pos_upper_lower_rel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    model_path: str = DEFAULT_ANKLE_MODEL_PATH,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos, joint_names = _selected_joint_data(asset, asset.data.joint_pos, asset_cfg)
    default_joint_pos, _ = _selected_joint_data(asset, asset.data.default_joint_pos, asset_cfg)
    joint_pos_ul = _replace_pitch_roll_with_upper_lower(joint_pos, joint_names, model_path, "pos")
    default_joint_pos_ul = _replace_pitch_roll_with_upper_lower(default_joint_pos, joint_names, model_path, "pos")
    return joint_pos_ul - default_joint_pos_ul


def joint_vel_upper_lower(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    model_path: str = DEFAULT_ANKLE_MODEL_PATH,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel, joint_names = _selected_joint_data(asset, asset.data.joint_vel, asset_cfg)
    return _replace_pitch_roll_with_upper_lower(joint_vel, joint_names, model_path, "vel")


def joint_vel_upper_lower_rel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    model_path: str = DEFAULT_ANKLE_MODEL_PATH,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel, joint_names = _selected_joint_data(asset, asset.data.joint_vel, asset_cfg)
    default_joint_vel, _ = _selected_joint_data(asset, asset.data.default_joint_vel, asset_cfg)
    joint_vel_ul = _replace_pitch_roll_with_upper_lower(joint_vel, joint_names, model_path, "vel")
    default_joint_vel_ul = _replace_pitch_roll_with_upper_lower(default_joint_vel, joint_names, model_path, "vel")
    return joint_vel_ul - default_joint_vel_ul


def ref_joint_pos_upper_lower(
    env: ManagerBasedAnimationEnv,
    animation: str,
    flatten_steps_dim: bool = True,
    model_path: str = DEFAULT_ANKLE_MODEL_PATH,
    source_joint_names: Sequence[str] | None = None,
    output_joint_names: Sequence[str] | None = None,
    output_joint_signs: dict[str, float] | None = None,
) -> torch.Tensor:
    animation_term = env.animation_manager.get_term(animation)
    ref_dof_pos = animation_term.get_dof_pos()
    ref_joint_names = _reference_joint_names(env, ref_dof_pos.shape[-1], source_joint_names)
    if output_joint_names is None:
        ref_dof_pos = _replace_pitch_roll_with_upper_lower(ref_dof_pos, ref_joint_names, model_path, "pos")
    else:
        ref_dof_pos = _convert_named_pitch_roll_to_upper_lower(
            ref_dof_pos,
            ref_joint_names,
            output_joint_names,
            model_path,
            "pos",
            output_joint_signs,
        )
    if flatten_steps_dim:
        return ref_dof_pos.reshape(env.num_envs, -1)
    return ref_dof_pos


def ref_joint_vel_upper_lower(
    env: ManagerBasedAnimationEnv,
    animation: str,
    flatten_steps_dim: bool = True,
    model_path: str = DEFAULT_ANKLE_MODEL_PATH,
    source_joint_names: Sequence[str] | None = None,
    output_joint_names: Sequence[str] | None = None,
    output_joint_signs: dict[str, float] | None = None,
) -> torch.Tensor:
    animation_term = env.animation_manager.get_term(animation)
    ref_dof_vel = animation_term.get_dof_vel()
    ref_joint_names = _reference_joint_names(env, ref_dof_vel.shape[-1], source_joint_names)
    if output_joint_names is None:
        ref_dof_vel = _replace_pitch_roll_with_upper_lower(ref_dof_vel, ref_joint_names, model_path, "vel")
    else:
        ref_dof_vel = _convert_named_pitch_roll_to_upper_lower(
            ref_dof_vel,
            ref_joint_names,
            output_joint_names,
            model_path,
            "vel",
            output_joint_signs,
        )
    if flatten_steps_dim:
        return ref_dof_vel.reshape(env.num_envs, -1)
    return ref_dof_vel


def _reference_joint_names(
    env: ManagerBasedAnimationEnv,
    num_dofs: int,
    source_joint_names: Sequence[str] | None,
) -> Sequence[str]:
    if source_joint_names is not None:
        if len(source_joint_names) != num_dofs:
            raise ValueError(f"source_joint_names has {len(source_joint_names)} names, but reference has {num_dofs} DoFs")
        return source_joint_names

    robot_joint_names = env.scene["robot"].joint_names
    if len(robot_joint_names) == num_dofs:
        return robot_joint_names
    if num_dofs == len(LENS110_MOTION_JOINT_NAMES):
        return LENS110_MOTION_JOINT_NAMES
    raise ValueError(
        f"Cannot infer reference joint names for {num_dofs} DoFs from robot with {len(robot_joint_names)} joints."
    )


def _convert_named_pitch_roll_to_upper_lower(
    joint_data: torch.Tensor,
    source_joint_names: Sequence[str],
    output_joint_names: Sequence[str],
    model_path: str,
    kind: str,
    output_joint_signs: dict[str, float] | None = None,
) -> torch.Tensor:
    source_index = {name: index for index, name in enumerate(source_joint_names)}
    if any(name not in source_index for name in PITCH_ROLL_JOINT_NAMES):
        missing = [name for name in PITCH_ROLL_JOINT_NAMES if name not in source_index]
        raise ValueError(f"Reference joint data is missing pitch/roll joints required for conversion: {missing}")

    pr_values = joint_data[..., [source_index[name] for name in PITCH_ROLL_JOINT_NAMES]]
    converter = get_ankle_converter(model_path, pr_values.device, pr_values.dtype)
    if kind == "pos":
        ul_values = converter.pos_pr2ul(pr_values)
    else:
        zero_values = torch.zeros_like(pr_values)
        ul_values = converter.vel_pr2ul(pr_values) - converter.vel_pr2ul(zero_values)

    values_by_name = {name: joint_data[..., index] for name, index in source_index.items()}
    values_by_name.update({name: ul_values[..., index] for index, name in enumerate(UPPER_LOWER_JOINT_NAMES)})

    missing_outputs = [name for name in output_joint_names if name not in values_by_name]
    if missing_outputs:
        raise ValueError(f"Cannot build reference joint data for unknown output joints: {missing_outputs}")
    output = torch.stack([values_by_name[name] for name in output_joint_names], dim=-1)
    if output_joint_signs:
        signs = torch.tensor(
            [output_joint_signs.get(name, 1.0) for name in output_joint_names],
            device=output.device,
            dtype=output.dtype,
        )
        output = output * signs
    return output


def _selected_joint_data(
    asset: Articulation,
    joint_data: torch.Tensor,
    asset_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, list[str]]:
    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, slice):
        ids = list(range(asset.num_joints))[joint_ids]
    elif joint_ids is None:
        ids = list(range(asset.num_joints))
    else:
        ids = list(joint_ids)
    return joint_data[:, ids], [asset.joint_names[joint_id] for joint_id in ids]


def _replace_pitch_roll_with_upper_lower(
    joint_data: torch.Tensor,
    joint_names: Sequence[str],
    model_path: str,
    kind: str,
) -> torch.Tensor:
    name_to_index = {name: index for index, name in enumerate(joint_names)}
    if any(name not in name_to_index for name in PITCH_ROLL_JOINT_NAMES):
        return joint_data

    pr_indices = [name_to_index[name] for name in PITCH_ROLL_JOINT_NAMES]
    pr_values = joint_data[..., pr_indices]
    converter = get_ankle_converter(model_path, pr_values.device, pr_values.dtype)
    if kind == "pos":
        ul_values = converter.pos_pr2ul(pr_values)
    else:
        zero_values = torch.zeros_like(pr_values)
        ul_values = converter.vel_pr2ul(pr_values) - converter.vel_pr2ul(zero_values)

    joint_data_ul = joint_data.clone()
    joint_data_ul[..., pr_indices] = ul_values
    return joint_data_ul


class Lens110UpperLowerJointPositionAction(ActionTerm):
    """Accept upper/lower ankle actions and convert them to pitch/roll targets for the URDF robot."""

    cfg: Lens110UpperLowerJointPositionActionCfg

    def __init__(self, cfg: Lens110UpperLowerJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names, preserve_order=self.cfg.preserve_order
        )
        self._num_joints = len(self._joint_ids)
        self._action_names = list(self._joint_names)
        for pr_name, ul_name in zip(PITCH_ROLL_JOINT_NAMES, UPPER_LOWER_JOINT_NAMES):
            self._action_names[self._action_names.index(pr_name)] = ul_name

        self._ankle_indices = [self._joint_names.index(name) for name in PITCH_ROLL_JOINT_NAMES]
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._policy_actions = torch.zeros_like(self._raw_actions)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._ankle_pr_bias = torch.zeros(self.num_envs, len(self._ankle_indices), device=self.device)

        self._scale = self._parse_name_values(self.cfg.scale, default=1.0)
        self._offset = self._parse_name_values(self.cfg.offset, default=0.0)
        if self.cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
            default_pr = self._offset[:, self._ankle_indices]
            converter = get_ankle_converter(self.cfg.model_path, default_pr.device, default_pr.dtype)
            default_ul = converter.pos_pr2ul(default_pr)
            self._ankle_pr_bias = default_pr - converter.pos_ul2pr(default_ul)
            self._offset[:, self._ankle_indices] = default_ul
        self._clip = self._parse_clip()

    @property
    def action_dim(self) -> int:
        return self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._policy_actions = self._raw_actions * self._scale + self._offset
        if self._clip is not None:
            self._policy_actions = torch.clamp(
                self._policy_actions,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )

        self._processed_actions = self._policy_actions.clone()
        ankle_ul = self._policy_actions[:, self._ankle_indices]
        ankle_pr = get_ankle_converter(self.cfg.model_path, ankle_ul.device, ankle_ul.dtype).pos_ul2pr(ankle_ul)
        ankle_pr = ankle_pr + self._ankle_pr_bias
        self._processed_actions[:, self._ankle_indices] = ankle_pr

    def apply_actions(self):
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0

    def _parse_name_values(self, values: float | dict[str, float], default: float) -> float | torch.Tensor:
        if isinstance(values, (float, int)):
            return float(values)
        if isinstance(values, dict):
            parsed = torch.full((self.num_envs, self.action_dim), default, device=self.device)
            index_list, _, value_list = string_utils.resolve_matching_names_values(values, self._action_names)
            parsed[:, index_list] = torch.tensor(value_list, device=self.device)
            return parsed
        raise ValueError(f"Unsupported action value type: {type(values)}")

    def _parse_clip(self) -> torch.Tensor | None:
        if self.cfg.clip is None:
            return None
        if not isinstance(self.cfg.clip, dict):
            raise ValueError(f"Unsupported clip type: {type(self.cfg.clip)}. Supported type is dict.")
        clip = torch.tensor([[-float("inf"), float("inf")]], device=self.device).repeat(self.num_envs, self.action_dim, 1)
        index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.clip, self._action_names)
        clip[:, index_list] = torch.tensor(value_list, device=self.device)
        return clip

@configclass
class Lens110UpperLowerJointPositionActionCfg(ActionTermCfg):
    """Joint-position action that exposes ankle upper/lower commands to the policy."""

    class_type: type[ActionTerm] = Lens110UpperLowerJointPositionAction
    joint_names: list[str] = MISSING
    scale: float | dict[str, float] = 1.0
    offset: float | dict[str, float] = 0.0
    preserve_order: bool = False
    use_default_offset: bool = True
    model_path: str = DEFAULT_ANKLE_MODEL_PATH
