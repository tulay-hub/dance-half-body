from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from pathlib import Path
from typing import TYPE_CHECKING, Any
import importlib.util
import sys

import numpy as np
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


def _default_xml_tendon_solver_path() -> str:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "pitchRoll2UpperLower" / "solve" / "xml_tendon_pr_to_ul.py"
        if candidate.exists():
            return str(candidate)
    return str(
        Path(__file__).resolve().parents[8] / "pitchRoll2UpperLower" / "solve" / "xml_tendon_pr_to_ul.py"
    )


DEFAULT_XML_TENDON_SOLVER_PATH = _default_xml_tendon_solver_path()
DEFAULT_XML_TENDON_XML_PATH = ""


def _default_xml_tendon_mlp_model_dir() -> str:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "pitchRoll2UpperLower" / "solve" / "models" / "xml_tendon_mlp"
        if candidate.exists():
            return str(candidate)
    return str(Path(__file__).resolve().parents[8] / "pitchRoll2UpperLower" / "solve" / "models" / "xml_tendon_mlp")


DEFAULT_XML_TENDON_MLP_MODEL_DIR = _default_xml_tendon_mlp_model_dir()
_SOLVER_MODULE_CACHE: dict[str, Any] = {}
_SOLVER_CACHE: dict[tuple[str, str, int, float], Any] = {}
_LOOKUP_CACHE: dict[tuple[str, str, int, float, int, int], "AnkleXmlTendonLookupConverter"] = {}
_MLP_CACHE: dict[tuple[str, str], "AnkleXmlTendonMlpConverter"] = {}


def _load_xml_tendon_solver_module(solver_path: str = DEFAULT_XML_TENDON_SOLVER_PATH) -> Any:
    path = str(Path(solver_path).expanduser().resolve())
    if path in _SOLVER_MODULE_CACHE:
        return _SOLVER_MODULE_CACHE[path]
    spec = importlib.util.spec_from_file_location("lens110_xml_tendon_pr_to_ul", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load XML tendon solver from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _SOLVER_MODULE_CACHE[path] = module
    return module


def get_xml_tendon_solver(
    solver_path: str = DEFAULT_XML_TENDON_SOLVER_PATH,
    xml_path: str = DEFAULT_XML_TENDON_XML_PATH,
    grid_samples: int = 801,
    length_tol: float = 1e-8,
):
    module = _load_xml_tendon_solver_module(solver_path)
    resolved_solver_path = str(Path(solver_path).expanduser().resolve())
    resolved_xml_path = "" if not xml_path else str(Path(xml_path).expanduser().resolve())
    cache_key = (resolved_solver_path, resolved_xml_path, int(grid_samples), float(length_tol))
    if cache_key not in _SOLVER_CACHE:
        kwargs = {"grid_samples": int(grid_samples), "length_tol": float(length_tol)}
        if resolved_xml_path:
            kwargs["xml_path"] = resolved_xml_path
        _SOLVER_CACHE[cache_key] = module.Lens110XmlTendonSolver(**kwargs)
    return _SOLVER_CACHE[cache_key]


class AnkleXmlTendonMlpConverter:
    """TorchScript evaluator for XML-tendon-trained PR/UL state relation models."""

    def __init__(self, model_dir: str, device: torch.device):
        model_root = Path(model_dir)
        self._pr_to_ul_state = torch.jit.load(
            str(model_root / "pr_to_ul_state" / "scripted.pth"), map_location=device
        ).eval()
        self._ul_to_pr_state = torch.jit.load(
            str(model_root / "ul_to_pr_state" / "scripted.pth"), map_location=device
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


def get_xml_tendon_mlp_converter(
    model_dir: str = DEFAULT_XML_TENDON_MLP_MODEL_DIR,
    device: torch.device | str = "cpu",
) -> AnkleXmlTendonMlpConverter:
    device = torch.device(device)
    cache_key = (str(Path(model_dir).resolve()), str(device))
    if cache_key not in _MLP_CACHE:
        _MLP_CACHE[cache_key] = AnkleXmlTendonMlpConverter(cache_key[0], device)
    return _MLP_CACHE[cache_key]


class AnkleXmlTendonLookupConverter:
    """Vectorized PR/UL converter sampled from the XML tendon solver."""

    def __init__(
        self,
        solver_path: str = DEFAULT_XML_TENDON_SOLVER_PATH,
        xml_path: str = DEFAULT_XML_TENDON_XML_PATH,
        grid_samples: int = 801,
        length_tol: float = 1e-8,
        lookup_pitch_samples: int = 81,
        lookup_roll_samples: int = 41,
    ):
        from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, RegularGridInterpolator

        self.solver = get_xml_tendon_solver(solver_path, xml_path, grid_samples, length_tol)
        self._RegularGridInterpolator = RegularGridInterpolator
        self._LinearNDInterpolator = LinearNDInterpolator
        self._NearestNDInterpolator = NearestNDInterpolator
        self._eps = 1.0e-4

        self._left = self._build_side_lookup(
            "left",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            int(lookup_pitch_samples),
            int(lookup_roll_samples),
        )
        self._right = self._build_side_lookup(
            "right",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            int(lookup_pitch_samples),
            int(lookup_roll_samples),
        )

    def _build_side_lookup(
        self,
        side: str,
        pitch_joint_name: str,
        roll_joint_name: str,
        lookup_pitch_samples: int,
        lookup_roll_samples: int,
    ) -> dict[str, Any]:
        pitch_lo, pitch_hi = self.solver.joint_range[pitch_joint_name]
        roll_lo, roll_hi = self.solver.joint_range[roll_joint_name]
        pitches = np.linspace(pitch_lo, pitch_hi, lookup_pitch_samples, dtype=np.float64)
        rolls = np.linspace(roll_lo, roll_hi, lookup_roll_samples, dtype=np.float64)

        ul_grid = np.zeros((len(pitches), len(rolls), 2), dtype=np.float64)
        last = None
        for i, pitch in enumerate(pitches):
            for j, roll in enumerate(rolls):
                ul_grid[i, j] = self.solver.solve_side(side, float(pitch), float(roll), initial=last)
                last = ul_grid[i, j]

        pr_points = np.stack(np.meshgrid(pitches, rolls, indexing="ij"), axis=-1).reshape(-1, 2)
        ul_points = ul_grid.reshape(-1, 2)
        ul_points, unique_indices = np.unique(ul_points, axis=0, return_index=True)
        pr_points = pr_points[unique_indices]
        return {
            "pitch_range": (float(pitch_lo), float(pitch_hi)),
            "roll_range": (float(roll_lo), float(roll_hi)),
            "pr_to_ul": [
                self._RegularGridInterpolator((pitches, rolls), ul_grid[..., 0], bounds_error=False, fill_value=None),
                self._RegularGridInterpolator((pitches, rolls), ul_grid[..., 1], bounds_error=False, fill_value=None),
            ],
            "ul_to_pr_linear": [
                self._LinearNDInterpolator(ul_points, pr_points[:, 0], fill_value=np.nan),
                self._LinearNDInterpolator(ul_points, pr_points[:, 1], fill_value=np.nan),
            ],
            "ul_to_pr_nearest": [
                self._NearestNDInterpolator(ul_points, pr_points[:, 0]),
                self._NearestNDInterpolator(ul_points, pr_points[:, 1]),
            ],
        }

    @staticmethod
    def _eval_regular(interpolators: list[Any], points: np.ndarray) -> np.ndarray:
        return np.column_stack([interpolator(points) for interpolator in interpolators])

    @staticmethod
    def _eval_scattered(side_lookup: dict[str, Any], points: np.ndarray) -> np.ndarray:
        out = np.column_stack([interpolator(points) for interpolator in side_lookup["ul_to_pr_linear"]])
        missing = np.isnan(out).any(axis=1)
        if missing.any():
            out[missing] = np.column_stack(
                [interpolator(points[missing]) for interpolator in side_lookup["ul_to_pr_nearest"]]
            )
        pitch_lo, pitch_hi = side_lookup["pitch_range"]
        roll_lo, roll_hi = side_lookup["roll_range"]
        out[:, 0] = np.clip(out[:, 0], pitch_lo, pitch_hi)
        out[:, 1] = np.clip(out[:, 1], roll_lo, roll_hi)
        return out

    def pr_to_ul_pos(self, pr_pos: np.ndarray) -> np.ndarray:
        pr_pos = np.asarray(pr_pos, dtype=np.float64).reshape(-1, 4)
        out = np.zeros_like(pr_pos)
        out[:, :2] = self._eval_regular(self._left["pr_to_ul"], pr_pos[:, :2])
        out[:, 2:4] = self._eval_regular(self._right["pr_to_ul"], pr_pos[:, 2:4])
        return out

    def pr_to_ul_vel(self, pr_pos: np.ndarray, pr_vel: np.ndarray) -> np.ndarray:
        pr_pos = np.asarray(pr_pos, dtype=np.float64).reshape(-1, 4)
        pr_vel = np.asarray(pr_vel, dtype=np.float64).reshape(-1, 4)
        out = np.zeros_like(pr_vel)
        for side_slice in (slice(0, 2), slice(2, 4)):
            for axis in range(2):
                plus = pr_pos.copy()
                minus = pr_pos.copy()
                plus[:, side_slice.start + axis] += self._eps
                minus[:, side_slice.start + axis] -= self._eps
                derivative = (self.pr_to_ul_pos(plus)[:, side_slice] - self.pr_to_ul_pos(minus)[:, side_slice]) / (
                    2.0 * self._eps
                )
                out[:, side_slice] += derivative * pr_vel[:, side_slice.start + axis : side_slice.start + axis + 1]
        return out

    def ul_to_pr_pos(self, ul_pos: np.ndarray) -> np.ndarray:
        ul_pos = np.asarray(ul_pos, dtype=np.float64).reshape(-1, 4)
        out = np.zeros_like(ul_pos)
        out[:, :2] = self._eval_scattered(self._left, ul_pos[:, :2])
        out[:, 2:4] = self._eval_scattered(self._right, ul_pos[:, 2:4])
        return out


def get_xml_tendon_lookup_converter(
    solver_path: str = DEFAULT_XML_TENDON_SOLVER_PATH,
    xml_path: str = DEFAULT_XML_TENDON_XML_PATH,
    grid_samples: int = 801,
    length_tol: float = 1e-8,
    lookup_pitch_samples: int = 81,
    lookup_roll_samples: int = 41,
) -> AnkleXmlTendonLookupConverter:
    resolved_solver_path = str(Path(solver_path).expanduser().resolve())
    resolved_xml_path = "" if not xml_path else str(Path(xml_path).expanduser().resolve())
    cache_key = (
        resolved_solver_path,
        resolved_xml_path,
        int(grid_samples),
        float(length_tol),
        int(lookup_pitch_samples),
        int(lookup_roll_samples),
    )
    if cache_key not in _LOOKUP_CACHE:
        _LOOKUP_CACHE[cache_key] = AnkleXmlTendonLookupConverter(
            resolved_solver_path,
            resolved_xml_path,
            grid_samples,
            length_tol,
            lookup_pitch_samples,
            lookup_roll_samples,
        )
    return _LOOKUP_CACHE[cache_key]


def joint_pos_xml_tendon_upper_lower(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    model_dir: str = DEFAULT_XML_TENDON_MLP_MODEL_DIR,
    solver_path: str = DEFAULT_XML_TENDON_SOLVER_PATH,
    xml_path: str = DEFAULT_XML_TENDON_XML_PATH,
    grid_samples: int = 801,
    length_tol: float = 1e-8,
    lookup_pitch_samples: int = 81,
    lookup_roll_samples: int = 41,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos, joint_names = _selected_joint_data(asset, asset.data.joint_pos, asset_cfg)
    return _replace_pitch_roll_with_upper_lower_xml(
        joint_pos,
        joint_names,
        solver_path,
        xml_path,
        grid_samples,
        length_tol,
        "pos",
        model_dir=model_dir,
        lookup_pitch_samples=lookup_pitch_samples,
        lookup_roll_samples=lookup_roll_samples,
    )


def joint_pos_xml_tendon_upper_lower_rel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    model_dir: str = DEFAULT_XML_TENDON_MLP_MODEL_DIR,
    solver_path: str = DEFAULT_XML_TENDON_SOLVER_PATH,
    xml_path: str = DEFAULT_XML_TENDON_XML_PATH,
    grid_samples: int = 801,
    length_tol: float = 1e-8,
    lookup_pitch_samples: int = 81,
    lookup_roll_samples: int = 41,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos, joint_names = _selected_joint_data(asset, asset.data.joint_pos, asset_cfg)
    default_joint_pos, _ = _selected_joint_data(asset, asset.data.default_joint_pos, asset_cfg)
    joint_pos_ul = _replace_pitch_roll_with_upper_lower_xml(
        joint_pos,
        joint_names,
        solver_path,
        xml_path,
        grid_samples,
        length_tol,
        "pos",
        model_dir=model_dir,
        lookup_pitch_samples=lookup_pitch_samples,
        lookup_roll_samples=lookup_roll_samples,
    )
    default_joint_pos_ul = _replace_pitch_roll_with_upper_lower_xml(
        default_joint_pos,
        joint_names,
        solver_path,
        xml_path,
        grid_samples,
        length_tol,
        "pos",
        model_dir=model_dir,
        lookup_pitch_samples=lookup_pitch_samples,
        lookup_roll_samples=lookup_roll_samples,
    )
    return joint_pos_ul - default_joint_pos_ul


def joint_vel_xml_tendon_upper_lower(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    model_dir: str = DEFAULT_XML_TENDON_MLP_MODEL_DIR,
    solver_path: str = DEFAULT_XML_TENDON_SOLVER_PATH,
    xml_path: str = DEFAULT_XML_TENDON_XML_PATH,
    grid_samples: int = 801,
    length_tol: float = 1e-8,
    lookup_pitch_samples: int = 81,
    lookup_roll_samples: int = 41,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos, joint_names = _selected_joint_data(asset, asset.data.joint_pos, asset_cfg)
    joint_vel, _ = _selected_joint_data(asset, asset.data.joint_vel, asset_cfg)
    return _replace_pitch_roll_with_upper_lower_xml(
        joint_pos,
        joint_names,
        solver_path,
        xml_path,
        grid_samples,
        length_tol,
        "vel",
        joint_vel=joint_vel,
        model_dir=model_dir,
        lookup_pitch_samples=lookup_pitch_samples,
        lookup_roll_samples=lookup_roll_samples,
    )


def joint_vel_xml_tendon_upper_lower_rel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    model_dir: str = DEFAULT_XML_TENDON_MLP_MODEL_DIR,
    solver_path: str = DEFAULT_XML_TENDON_SOLVER_PATH,
    xml_path: str = DEFAULT_XML_TENDON_XML_PATH,
    grid_samples: int = 801,
    length_tol: float = 1e-8,
    lookup_pitch_samples: int = 81,
    lookup_roll_samples: int = 41,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos, joint_names = _selected_joint_data(asset, asset.data.joint_pos, asset_cfg)
    joint_vel, _ = _selected_joint_data(asset, asset.data.joint_vel, asset_cfg)
    default_joint_pos, _ = _selected_joint_data(asset, asset.data.default_joint_pos, asset_cfg)
    default_joint_vel, _ = _selected_joint_data(asset, asset.data.default_joint_vel, asset_cfg)
    joint_vel_ul = _replace_pitch_roll_with_upper_lower_xml(
        joint_pos,
        joint_names,
        solver_path,
        xml_path,
        grid_samples,
        length_tol,
        "vel",
        joint_vel=joint_vel,
        model_dir=model_dir,
        lookup_pitch_samples=lookup_pitch_samples,
        lookup_roll_samples=lookup_roll_samples,
    )
    default_joint_vel_ul = _replace_pitch_roll_with_upper_lower_xml(
        default_joint_pos,
        joint_names,
        solver_path,
        xml_path,
        grid_samples,
        length_tol,
        "vel",
        joint_vel=default_joint_vel,
        model_dir=model_dir,
        lookup_pitch_samples=lookup_pitch_samples,
        lookup_roll_samples=lookup_roll_samples,
    )
    return joint_vel_ul - default_joint_vel_ul


def ref_joint_pos_xml_tendon_upper_lower(
    env: ManagerBasedAnimationEnv,
    animation: str,
    flatten_steps_dim: bool = True,
    model_dir: str = DEFAULT_XML_TENDON_MLP_MODEL_DIR,
    solver_path: str = DEFAULT_XML_TENDON_SOLVER_PATH,
    xml_path: str = DEFAULT_XML_TENDON_XML_PATH,
    grid_samples: int = 801,
    length_tol: float = 1e-8,
    lookup_pitch_samples: int = 81,
    lookup_roll_samples: int = 41,
    source_joint_names: Sequence[str] | None = None,
    output_joint_names: Sequence[str] | None = None,
    output_joint_signs: dict[str, float] | None = None,
) -> torch.Tensor:
    animation_term = env.animation_manager.get_term(animation)
    ref_dof_pos = animation_term.get_dof_pos()
    ref_joint_names = _reference_joint_names(env, ref_dof_pos.shape[-1], source_joint_names)
    if output_joint_names is None:
        ref_dof_pos = _replace_pitch_roll_with_upper_lower_xml(
            ref_dof_pos,
            ref_joint_names,
            solver_path,
            xml_path,
            grid_samples,
            length_tol,
            "pos",
            model_dir=model_dir,
            lookup_pitch_samples=lookup_pitch_samples,
            lookup_roll_samples=lookup_roll_samples,
        )
    else:
        ref_dof_pos = _convert_named_pitch_roll_to_upper_lower_xml(
            ref_dof_pos,
            ref_joint_names,
            output_joint_names,
            model_dir,
            solver_path,
            xml_path,
            grid_samples,
            length_tol,
            lookup_pitch_samples,
            lookup_roll_samples,
            "pos",
            output_joint_signs,
        )
    if flatten_steps_dim:
        return ref_dof_pos.reshape(env.num_envs, -1)
    return ref_dof_pos


def ref_joint_vel_xml_tendon_upper_lower(
    env: ManagerBasedAnimationEnv,
    animation: str,
    flatten_steps_dim: bool = True,
    model_dir: str = DEFAULT_XML_TENDON_MLP_MODEL_DIR,
    solver_path: str = DEFAULT_XML_TENDON_SOLVER_PATH,
    xml_path: str = DEFAULT_XML_TENDON_XML_PATH,
    grid_samples: int = 801,
    length_tol: float = 1e-8,
    lookup_pitch_samples: int = 81,
    lookup_roll_samples: int = 41,
    source_joint_names: Sequence[str] | None = None,
    output_joint_names: Sequence[str] | None = None,
    output_joint_signs: dict[str, float] | None = None,
) -> torch.Tensor:
    animation_term = env.animation_manager.get_term(animation)
    ref_dof_pos = animation_term.get_dof_pos()
    ref_dof_vel = animation_term.get_dof_vel()
    ref_joint_names = _reference_joint_names(env, ref_dof_vel.shape[-1], source_joint_names)
    if output_joint_names is None:
        ref_dof_vel = _replace_pitch_roll_with_upper_lower_xml(
            ref_dof_pos,
            ref_joint_names,
            solver_path,
            xml_path,
            grid_samples,
            length_tol,
            "vel",
            joint_vel=ref_dof_vel,
            model_dir=model_dir,
            lookup_pitch_samples=lookup_pitch_samples,
            lookup_roll_samples=lookup_roll_samples,
        )
    else:
        ref_dof_vel = _convert_named_pitch_roll_to_upper_lower_xml(
            ref_dof_pos,
            ref_joint_names,
            output_joint_names,
            model_dir,
            solver_path,
            xml_path,
            grid_samples,
            length_tol,
            lookup_pitch_samples,
            lookup_roll_samples,
            "vel",
            output_joint_signs,
            joint_vel=ref_dof_vel,
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


def _convert_named_pitch_roll_to_upper_lower_xml(
    joint_pos: torch.Tensor,
    source_joint_names: Sequence[str],
    output_joint_names: Sequence[str],
    model_dir: str,
    solver_path: str,
    xml_path: str,
    grid_samples: int,
    length_tol: float,
    lookup_pitch_samples: int,
    lookup_roll_samples: int,
    kind: str,
    output_joint_signs: dict[str, float] | None = None,
    joint_vel: torch.Tensor | None = None,
) -> torch.Tensor:
    source_index = {name: index for index, name in enumerate(source_joint_names)}
    if any(name not in source_index for name in PITCH_ROLL_JOINT_NAMES):
        missing = [name for name in PITCH_ROLL_JOINT_NAMES if name not in source_index]
        raise ValueError(f"Reference joint data is missing pitch/roll joints required for conversion: {missing}")

    pr_pos = joint_pos[..., [source_index[name] for name in PITCH_ROLL_JOINT_NAMES]]
    if kind == "pos":
        ul_values = _pr_to_ul_pos_tensor(
            pr_pos,
            solver_path,
            xml_path,
            grid_samples,
            length_tol,
            model_dir,
            lookup_pitch_samples,
            lookup_roll_samples,
        )
        source_data = joint_pos
    else:
        if joint_vel is None:
            raise ValueError("joint_vel is required for XML tendon velocity conversion")
        pr_vel = joint_vel[..., [source_index[name] for name in PITCH_ROLL_JOINT_NAMES]]
        ul_values = _pr_to_ul_vel_tensor(
            pr_pos,
            pr_vel,
            solver_path,
            xml_path,
            grid_samples,
            length_tol,
            model_dir,
            lookup_pitch_samples,
            lookup_roll_samples,
        )
        source_data = joint_vel

    values_by_name = {name: source_data[..., index] for name, index in source_index.items()}
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


def _replace_pitch_roll_with_upper_lower_xml(
    joint_pos: torch.Tensor,
    joint_names: Sequence[str],
    solver_path: str,
    xml_path: str,
    grid_samples: int,
    length_tol: float,
    kind: str,
    joint_vel: torch.Tensor | None = None,
    model_dir: str = DEFAULT_XML_TENDON_MLP_MODEL_DIR,
    lookup_pitch_samples: int = 81,
    lookup_roll_samples: int = 41,
) -> torch.Tensor:
    name_to_index = {name: index for index, name in enumerate(joint_names)}
    if any(name not in name_to_index for name in PITCH_ROLL_JOINT_NAMES):
        return joint_pos if kind == "pos" else joint_vel

    pr_indices = [name_to_index[name] for name in PITCH_ROLL_JOINT_NAMES]
    pr_pos = joint_pos[..., pr_indices]
    if kind == "pos":
        ul_values = _pr_to_ul_pos_tensor(
            pr_pos,
            solver_path,
            xml_path,
            grid_samples,
            length_tol,
            model_dir,
            lookup_pitch_samples,
            lookup_roll_samples,
        )
        joint_data_ul = joint_pos.clone()
    else:
        if joint_vel is None:
            raise ValueError("joint_vel is required for XML tendon velocity conversion")
        pr_vel = joint_vel[..., pr_indices]
        ul_values = _pr_to_ul_vel_tensor(
            pr_pos,
            pr_vel,
            solver_path,
            xml_path,
            grid_samples,
            length_tol,
            model_dir,
            lookup_pitch_samples,
            lookup_roll_samples,
        )
        joint_data_ul = joint_vel.clone()
    joint_data_ul[..., pr_indices] = ul_values
    return joint_data_ul


def _pr_to_ul_pos_tensor(
    pr_pos: torch.Tensor,
    solver_path: str,
    xml_path: str,
    grid_samples: int,
    length_tol: float,
    model_dir: str = DEFAULT_XML_TENDON_MLP_MODEL_DIR,
    lookup_pitch_samples: int = 81,
    lookup_roll_samples: int = 41,
) -> torch.Tensor:
    del solver_path, xml_path, grid_samples, length_tol, lookup_pitch_samples, lookup_roll_samples
    converter = get_xml_tendon_mlp_converter(model_dir, pr_pos.device)
    zero_vel = torch.zeros_like(pr_pos)
    ul_pos, _ = converter.pr_to_ul_state(pr_pos, zero_vel)
    return ul_pos


def _pr_to_ul_vel_tensor(
    pr_pos: torch.Tensor,
    pr_vel: torch.Tensor,
    solver_path: str,
    xml_path: str,
    grid_samples: int,
    length_tol: float,
    model_dir: str = DEFAULT_XML_TENDON_MLP_MODEL_DIR,
    lookup_pitch_samples: int = 81,
    lookup_roll_samples: int = 41,
) -> torch.Tensor:
    del solver_path, xml_path, grid_samples, length_tol, lookup_pitch_samples, lookup_roll_samples
    converter = get_xml_tendon_mlp_converter(model_dir, pr_pos.device)
    _, ul_vel = converter.pr_to_ul_state(pr_pos, pr_vel)
    return ul_vel


def _ul_to_pr_pos_tensor(
    ul_pos: torch.Tensor,
    solver_path: str,
    xml_path: str,
    grid_samples: int,
    length_tol: float,
    model_dir: str = DEFAULT_XML_TENDON_MLP_MODEL_DIR,
    lookup_pitch_samples: int = 81,
    lookup_roll_samples: int = 41,
) -> torch.Tensor:
    del solver_path, xml_path, grid_samples, length_tol, lookup_pitch_samples, lookup_roll_samples
    converter = get_xml_tendon_mlp_converter(model_dir, ul_pos.device)
    zero_vel = torch.zeros_like(ul_pos)
    pr_pos, _ = converter.ul_to_pr_state(ul_pos, zero_vel)
    return pr_pos


class Lens110XmlTendonUpperLowerJointPositionAction(ActionTerm):
    """Accept upper/lower ankle actions and convert them to pitch/roll targets with the XML tendon solver."""

    cfg: Lens110XmlTendonUpperLowerJointPositionActionCfg

    def __init__(self, cfg: Lens110XmlTendonUpperLowerJointPositionActionCfg, env: ManagerBasedEnv):
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
            default_ul = _pr_to_ul_pos_tensor(
                default_pr,
                self.cfg.solver_path,
                self.cfg.xml_path,
                self.cfg.grid_samples,
                self.cfg.length_tol,
                self.cfg.model_dir,
                self.cfg.lookup_pitch_samples,
                self.cfg.lookup_roll_samples,
            )
            recovered_pr = _ul_to_pr_pos_tensor(
                default_ul,
                self.cfg.solver_path,
                self.cfg.xml_path,
                self.cfg.grid_samples,
                self.cfg.length_tol,
                self.cfg.model_dir,
                self.cfg.lookup_pitch_samples,
                self.cfg.lookup_roll_samples,
            )
            self._ankle_pr_bias = default_pr - recovered_pr
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
        ankle_pr = _ul_to_pr_pos_tensor(
            ankle_ul,
            self.cfg.solver_path,
            self.cfg.xml_path,
            self.cfg.grid_samples,
            self.cfg.length_tol,
            self.cfg.model_dir,
            self.cfg.lookup_pitch_samples,
            self.cfg.lookup_roll_samples,
        )
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
class Lens110XmlTendonUpperLowerJointPositionActionCfg(ActionTermCfg):
    """Joint-position action that exposes XML-tendon upper/lower ankle commands to the policy."""

    class_type: type[ActionTerm] = Lens110XmlTendonUpperLowerJointPositionAction
    joint_names: list[str] = MISSING
    scale: float | dict[str, float] = 1.0
    offset: float | dict[str, float] = 0.0
    clip: dict[str, tuple[float, float]] | None = None
    preserve_order: bool = False
    use_default_offset: bool = True
    model_dir: str = DEFAULT_XML_TENDON_MLP_MODEL_DIR
    solver_path: str = DEFAULT_XML_TENDON_SOLVER_PATH
    xml_path: str = DEFAULT_XML_TENDON_XML_PATH
    grid_samples: int = 801
    length_tol: float = 1e-8
    lookup_pitch_samples: int = 81
    lookup_roll_samples: int = 41
