from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from pathlib import Path
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.string as string_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from .ankle_upper_lower import (
    LENS110_MOTION_JOINT_NAMES,
    PITCH_ROLL_JOINT_NAMES,
    UPPER_LOWER_JOINT_NAMES,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from legged_lab.envs import ManagerBasedAnimationEnv


def _default_weighted_mlp_model_dir() -> str:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "pitchRoll2UpperLower" / "t_p_v" / "models" / "weighted_mlp"
        if candidate.exists():
            return str(candidate)
    return str(Path(__file__).resolve().parents[7] / "pitchRoll2UpperLower" / "t_p_v" / "models" / "weighted_mlp")


DEFAULT_WEIGHTED_MLP_MODEL_DIR = _default_weighted_mlp_model_dir()
_MLP_CACHE: dict[tuple[str, str], "AnkleWeightedMlpConverter"] = {}


class AnkleWeightedMlpConverter:
    """TorchScript evaluator for the weighted MLP PR/UL ankle relation models."""

    def __init__(self, model_dir: str, device: torch.device):
        model_root = Path(model_dir)
        self._pr_to_ul_state = torch.jit.load(
            str(model_root / "pr_to_ul_state" / "scripted.pth"), map_location=device
        ).eval()
        self._ul_to_pr_state = torch.jit.load(
            str(model_root / "ul_to_pr_state" / "scripted.pth"), map_location=device
        ).eval()
        self._pr_to_ul_torque = torch.jit.load(
            str(model_root / "pr_to_ul_torque" / "scripted.pth"), map_location=device
        ).eval()
        self._ul_to_pr_torque = torch.jit.load(
            str(model_root / "ul_to_pr_torque" / "scripted.pth"), map_location=device
        ).eval()

    @staticmethod
    def _run(model: torch.jit.ScriptModule, values: torch.Tensor) -> torch.Tensor:
        dtype = values.dtype
        with torch.no_grad():
            out = model(values.to(dtype=torch.float32))
        return out.to(dtype=dtype)

    def pr_to_ul_state(self, pr_pos: torch.Tensor, pr_vel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values_shape = pr_pos.shape
        x = torch.cat((pr_pos.reshape(-1, 4), pr_vel.reshape(-1, 4)), dim=-1)
        y = self._run(self._pr_to_ul_state, x).reshape(*values_shape[:-1], 8)
        return y[..., 0:4], y[..., 4:8]

    def ul_to_pr_state(self, ul_pos: torch.Tensor, ul_vel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values_shape = ul_pos.shape
        x = torch.cat((ul_pos.reshape(-1, 4), ul_vel.reshape(-1, 4)), dim=-1)
        y = self._run(self._ul_to_pr_state, x).reshape(*values_shape[:-1], 8)
        return y[..., 0:4], y[..., 4:8]

    def pr_to_ul_torque(
        self, pr_pos: torch.Tensor, pr_vel: torch.Tensor, pr_tau: torch.Tensor
    ) -> torch.Tensor:
        values_shape = pr_tau.shape
        x = torch.cat((pr_pos.reshape(-1, 4), pr_vel.reshape(-1, 4), pr_tau.reshape(-1, 4)), dim=-1)
        return self._run(self._pr_to_ul_torque, x).reshape(values_shape)

    def ul_to_pr_torque(
        self, ul_pos: torch.Tensor, ul_vel: torch.Tensor, ul_tau: torch.Tensor
    ) -> torch.Tensor:
        values_shape = ul_tau.shape
        x = torch.cat((ul_pos.reshape(-1, 4), ul_vel.reshape(-1, 4), ul_tau.reshape(-1, 4)), dim=-1)
        return self._run(self._ul_to_pr_torque, x).reshape(values_shape)


def get_weighted_mlp_ankle_converter(
    model_dir: str = DEFAULT_WEIGHTED_MLP_MODEL_DIR,
    device: torch.device | str = "cpu",
) -> AnkleWeightedMlpConverter:
    device = torch.device(device)
    cache_key = (str(Path(model_dir).resolve()), str(device))
    if cache_key not in _MLP_CACHE:
        _MLP_CACHE[cache_key] = AnkleWeightedMlpConverter(cache_key[0], device)
    return _MLP_CACHE[cache_key]


def joint_pos_upper_lower_mlp(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    model_dir: str = DEFAULT_WEIGHTED_MLP_MODEL_DIR,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos, joint_vel, joint_names = _selected_joint_state(asset, asset_cfg)
    joint_pos_ul, _ = _replace_pitch_roll_with_upper_lower_mlp(joint_pos, joint_vel, joint_names, model_dir)
    return joint_pos_ul


def joint_pos_upper_lower_mlp_rel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    model_dir: str = DEFAULT_WEIGHTED_MLP_MODEL_DIR,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos, joint_vel, joint_names = _selected_joint_state(asset, asset_cfg)
    default_joint_pos, default_joint_vel, _ = _selected_joint_state(asset, asset_cfg, default=True)
    joint_pos_ul, _ = _replace_pitch_roll_with_upper_lower_mlp(joint_pos, joint_vel, joint_names, model_dir)
    default_joint_pos_ul, _ = _replace_pitch_roll_with_upper_lower_mlp(
        default_joint_pos, default_joint_vel, joint_names, model_dir
    )
    return joint_pos_ul - default_joint_pos_ul


def joint_vel_upper_lower_mlp(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    model_dir: str = DEFAULT_WEIGHTED_MLP_MODEL_DIR,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos, joint_vel, joint_names = _selected_joint_state(asset, asset_cfg)
    _, joint_vel_ul = _replace_pitch_roll_with_upper_lower_mlp(joint_pos, joint_vel, joint_names, model_dir)
    return joint_vel_ul


def joint_vel_upper_lower_mlp_rel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    model_dir: str = DEFAULT_WEIGHTED_MLP_MODEL_DIR,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos, joint_vel, joint_names = _selected_joint_state(asset, asset_cfg)
    default_joint_pos, default_joint_vel, _ = _selected_joint_state(asset, asset_cfg, default=True)
    _, joint_vel_ul = _replace_pitch_roll_with_upper_lower_mlp(joint_pos, joint_vel, joint_names, model_dir)
    _, default_joint_vel_ul = _replace_pitch_roll_with_upper_lower_mlp(
        default_joint_pos, default_joint_vel, joint_names, model_dir
    )
    return joint_vel_ul - default_joint_vel_ul


def ref_joint_pos_upper_lower_mlp(
    env: ManagerBasedAnimationEnv,
    animation: str,
    flatten_steps_dim: bool = True,
    model_dir: str = DEFAULT_WEIGHTED_MLP_MODEL_DIR,
    source_joint_names: Sequence[str] | None = None,
    output_joint_names: Sequence[str] | None = None,
    output_joint_signs: dict[str, float] | None = None,
) -> torch.Tensor:
    animation_term = env.animation_manager.get_term(animation)
    ref_dof_pos = animation_term.get_dof_pos()
    ref_dof_vel = animation_term.get_dof_vel()
    ref_joint_names = _reference_joint_names(env, ref_dof_pos.shape[-1], source_joint_names)
    if output_joint_names is None:
        ref_dof_pos, _ = _replace_pitch_roll_with_upper_lower_mlp(ref_dof_pos, ref_dof_vel, ref_joint_names, model_dir)
    else:
        ref_dof_pos = _convert_named_pitch_roll_to_upper_lower_mlp(
            ref_dof_pos, ref_dof_vel, ref_joint_names, output_joint_names, model_dir, "pos", output_joint_signs
        )
    if flatten_steps_dim:
        return ref_dof_pos.reshape(env.num_envs, -1)
    return ref_dof_pos


def ref_joint_vel_upper_lower_mlp(
    env: ManagerBasedAnimationEnv,
    animation: str,
    flatten_steps_dim: bool = True,
    model_dir: str = DEFAULT_WEIGHTED_MLP_MODEL_DIR,
    source_joint_names: Sequence[str] | None = None,
    output_joint_names: Sequence[str] | None = None,
    output_joint_signs: dict[str, float] | None = None,
) -> torch.Tensor:
    animation_term = env.animation_manager.get_term(animation)
    ref_dof_pos = animation_term.get_dof_pos()
    ref_dof_vel = animation_term.get_dof_vel()
    ref_joint_names = _reference_joint_names(env, ref_dof_vel.shape[-1], source_joint_names)
    if output_joint_names is None:
        _, ref_dof_vel = _replace_pitch_roll_with_upper_lower_mlp(ref_dof_pos, ref_dof_vel, ref_joint_names, model_dir)
    else:
        ref_dof_vel = _convert_named_pitch_roll_to_upper_lower_mlp(
            ref_dof_pos, ref_dof_vel, ref_joint_names, output_joint_names, model_dir, "vel", output_joint_signs
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


def _convert_named_pitch_roll_to_upper_lower_mlp(
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    source_joint_names: Sequence[str],
    output_joint_names: Sequence[str],
    model_dir: str,
    kind: str,
    output_joint_signs: dict[str, float] | None = None,
) -> torch.Tensor:
    source_index = {name: index for index, name in enumerate(source_joint_names)}
    if any(name not in source_index for name in PITCH_ROLL_JOINT_NAMES):
        missing = [name for name in PITCH_ROLL_JOINT_NAMES if name not in source_index]
        raise ValueError(f"Reference joint data is missing pitch/roll joints required for conversion: {missing}")

    pr_pos = joint_pos[..., [source_index[name] for name in PITCH_ROLL_JOINT_NAMES]]
    pr_vel = joint_vel[..., [source_index[name] for name in PITCH_ROLL_JOINT_NAMES]]
    converter = get_weighted_mlp_ankle_converter(model_dir, pr_pos.device)
    ul_pos, ul_vel = converter.pr_to_ul_state(pr_pos, pr_vel)

    source_data = joint_pos if kind == "pos" else joint_vel
    values_by_name = {name: source_data[..., index] for name, index in source_index.items()}
    ul_values = ul_pos if kind == "pos" else ul_vel
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


def _selected_joint_state(
    asset: Articulation,
    asset_cfg: SceneEntityCfg,
    default: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, slice):
        ids = list(range(asset.num_joints))[joint_ids]
    elif joint_ids is None:
        ids = list(range(asset.num_joints))
    else:
        ids = list(joint_ids)

    if default:
        joint_pos = asset.data.default_joint_pos[:, ids]
        joint_vel = asset.data.default_joint_vel[:, ids]
    else:
        joint_pos = asset.data.joint_pos[:, ids]
        joint_vel = asset.data.joint_vel[:, ids]
    return joint_pos, joint_vel, [asset.joint_names[joint_id] for joint_id in ids]


def _replace_pitch_roll_with_upper_lower_mlp(
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    joint_names: Sequence[str],
    model_dir: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    name_to_index = {name: index for index, name in enumerate(joint_names)}
    if any(name not in name_to_index for name in PITCH_ROLL_JOINT_NAMES):
        return joint_pos, joint_vel

    pr_indices = [name_to_index[name] for name in PITCH_ROLL_JOINT_NAMES]
    pr_pos = joint_pos[..., pr_indices]
    pr_vel = joint_vel[..., pr_indices]
    converter = get_weighted_mlp_ankle_converter(model_dir, pr_pos.device)
    ul_pos, ul_vel = converter.pr_to_ul_state(pr_pos, pr_vel)

    joint_pos_ul = joint_pos.clone()
    joint_vel_ul = joint_vel.clone()
    joint_pos_ul[..., pr_indices] = ul_pos
    joint_vel_ul[..., pr_indices] = ul_vel
    return joint_pos_ul, joint_vel_ul


class Lens110MlpUpperLowerJointTorquePdAction(ActionTerm):
    """Expose upper/lower ankle actions and apply converted PR ankle torques to the URDF robot."""

    cfg: Lens110MlpUpperLowerJointTorquePdActionCfg

    def __init__(self, cfg: Lens110MlpUpperLowerJointTorquePdActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names, preserve_order=self.cfg.preserve_order
        )
        self._num_joints = len(self._joint_ids)
        self._action_names = list(self._joint_names)
        for pr_name, ul_name in zip(PITCH_ROLL_JOINT_NAMES, UPPER_LOWER_JOINT_NAMES):
            self._action_names[self._action_names.index(pr_name)] = ul_name

        self._ankle_indices = [self._joint_names.index(name) for name in PITCH_ROLL_JOINT_NAMES]
        self._non_ankle_indices = [index for index in range(self._num_joints) if index not in self._ankle_indices]
        self._ankle_joint_ids = [self._joint_ids[index] for index in self._ankle_indices]
        self._non_ankle_joint_ids = [self._joint_ids[index] for index in self._non_ankle_indices]

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._policy_actions = torch.zeros_like(self._raw_actions)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._ankle_efforts = torch.zeros(self.num_envs, len(self._ankle_indices), device=self.device)
        self._ankle_pr_bias = torch.zeros_like(self._ankle_efforts)

        self._scale = self._parse_name_values(self.cfg.scale, default=1.0)
        self._offset = self._parse_name_values(self.cfg.offset, default=0.0)
        if self.cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
            default_pr_pos = self._offset[:, self._ankle_indices]
            default_pr_vel = torch.zeros_like(default_pr_pos)
            converter = get_weighted_mlp_ankle_converter(self.cfg.model_dir, default_pr_pos.device)
            default_ul_pos, default_ul_vel = converter.pr_to_ul_state(default_pr_pos, default_pr_vel)
            recovered_pr_pos, _ = converter.ul_to_pr_state(default_ul_pos, default_ul_vel)
            self._ankle_pr_bias = default_pr_pos - recovered_pr_pos
            self._offset[:, self._ankle_indices] = default_ul_pos

        self._clip = self._parse_clip()
        self._ankle_stiffness = self._parse_name_values(self.cfg.ankle_stiffness, default=5.0)
        self._ankle_damping = self._parse_name_values(self.cfg.ankle_damping, default=5.0)
        if not isinstance(self._ankle_stiffness, torch.Tensor):
            self._ankle_stiffness = torch.full_like(self._ankle_efforts, float(self._ankle_stiffness))
        else:
            self._ankle_stiffness = self._ankle_stiffness[:, self._ankle_indices]
        if not isinstance(self._ankle_damping, torch.Tensor):
            self._ankle_damping = torch.full_like(self._ankle_efforts, float(self._ankle_damping))
        else:
            self._ankle_damping = self._ankle_damping[:, self._ankle_indices]

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
        converter = get_weighted_mlp_ankle_converter(self.cfg.model_dir, self.device)

        ankle_ul_target = self._policy_actions[:, self._ankle_indices]
        ankle_ul_target_vel = torch.zeros_like(ankle_ul_target)
        ankle_pr_target, _ = converter.ul_to_pr_state(ankle_ul_target, ankle_ul_target_vel)
        ankle_pr_target = ankle_pr_target + self._ankle_pr_bias
        self._processed_actions[:, self._ankle_indices] = ankle_pr_target

        ankle_pr_pos = self._asset.data.joint_pos[:, self._ankle_joint_ids]
        ankle_pr_vel = self._asset.data.joint_vel[:, self._ankle_joint_ids]
        if self.cfg.ankle_pd_space == "pitch_roll":
            ankle_pr_tau = self._ankle_stiffness * (ankle_pr_target - ankle_pr_pos) - self._ankle_damping * ankle_pr_vel
        elif self.cfg.ankle_pd_space == "mlp_torque_roundtrip":
            ankle_pr_tau_pd = self._ankle_stiffness * (ankle_pr_target - ankle_pr_pos) - self._ankle_damping * ankle_pr_vel
            ankle_ul_pos, ankle_ul_vel = converter.pr_to_ul_state(ankle_pr_pos, ankle_pr_vel)
            ankle_ul_tau = converter.pr_to_ul_torque(ankle_pr_pos, ankle_pr_vel, ankle_pr_tau_pd)
            ankle_pr_tau = converter.ul_to_pr_torque(ankle_ul_pos, ankle_ul_vel, ankle_ul_tau)
        elif self.cfg.ankle_pd_space == "upper_lower":
            ankle_ul_pos, ankle_ul_vel = converter.pr_to_ul_state(ankle_pr_pos, ankle_pr_vel)
            ankle_ul_tau = self._ankle_stiffness * (ankle_ul_target - ankle_ul_pos) - self._ankle_damping * ankle_ul_vel
            ankle_pr_tau = converter.ul_to_pr_torque(ankle_ul_pos, ankle_ul_vel, ankle_ul_tau)
        else:
            raise ValueError(f"Unsupported ankle_pd_space: {self.cfg.ankle_pd_space}")
        effort_limit = float(self.cfg.ankle_effort_limit)
        self._ankle_efforts = torch.clamp(ankle_pr_tau, min=-effort_limit, max=effort_limit)

    def apply_actions(self):
        if self._non_ankle_joint_ids:
            self._asset.set_joint_position_target(
                self._processed_actions[:, self._non_ankle_indices],
                joint_ids=self._non_ankle_joint_ids,
            )
        self._asset.set_joint_position_target(
            self._processed_actions[:, self._ankle_indices],
            joint_ids=self._ankle_joint_ids,
        )
        self._asset.set_joint_effort_target(self._ankle_efforts, joint_ids=self._ankle_joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0
        self._ankle_efforts[env_ids] = 0.0

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
class Lens110MlpUpperLowerJointTorquePdActionCfg(ActionTermCfg):
    """Joint action that exposes upper/lower ankle commands and uses MLP-converted ankle torque PD."""

    class_type: type[ActionTerm] = Lens110MlpUpperLowerJointTorquePdAction
    joint_names: list[str] = MISSING
    scale: float | dict[str, float] = 1.0
    offset: float | dict[str, float] = 0.0
    clip: dict[str, tuple[float, float]] | None = None
    preserve_order: bool = False
    use_default_offset: bool = True
    model_dir: str = DEFAULT_WEIGHTED_MLP_MODEL_DIR
    ankle_stiffness: float | dict[str, float] = 5.0
    ankle_damping: float | dict[str, float] = 5.0
    ankle_effort_limit: float = 36.0
    ankle_pd_space: str = "upper_lower"
