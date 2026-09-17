"""Solve Lens110 ankle pitch/roll <-> upper/lower from MJCF tendon geometry.

This does not fit a polynomial or MLP model.  It loads the Lens110 MJCF, fixes
the four spatial tendon lengths defined in the XML, and numerically solves the
ankle coordinates that make those lengths match for a requested pose.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_XML = (
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

PR_NAMES = (
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)
UL_NAMES = (
    "left_ankle_upper_joint",
    "left_ankle_lower_joint",
    "right_ankle_upper_joint",
    "right_ankle_lower_joint",
)


@dataclass(frozen=True)
class TendonChannel:
    side: str
    name: str
    pitch_joint: str
    roll_joint: str
    solve_joint: str
    tendon_id: int


CHANNELS = (
    TendonChannel("left", "upper", "left_ankle_pitch_joint", "left_ankle_roll_joint", "left_ankle_upper_joint", 0),
    TendonChannel("left", "lower", "left_ankle_pitch_joint", "left_ankle_roll_joint", "left_ankle_lower_joint", 1),
    TendonChannel("right", "upper", "right_ankle_pitch_joint", "right_ankle_roll_joint", "right_ankle_upper_joint", 2),
    TendonChannel("right", "lower", "right_ankle_pitch_joint", "right_ankle_roll_joint", "right_ankle_lower_joint", 3),
)


def _finite_diff(values: np.ndarray, dt: float) -> np.ndarray:
    vel = np.zeros_like(values)
    if len(values) <= 1:
        return vel
    vel[1:-1] = (values[2:] - values[:-2]) / (2.0 * dt)
    vel[0] = (values[1] - values[0]) / dt
    vel[-1] = (values[-1] - values[-2]) / dt
    return vel


def _as_scalar(value) -> float:
    return float(np.asarray(value).reshape(-1)[0])


class Lens110XmlTendonSolver:
    """Pitch/roll to upper/lower solver driven by MuJoCo tendon lengths."""

    def __init__(
        self,
        xml_path: str | Path = DEFAULT_XML,
        *,
        grid_samples: int = 801,
        length_tol: float = 1e-8,
    ) -> None:
        self.xml_path = Path(xml_path).expanduser().resolve()
        self.model = self._load_model(self.xml_path)
        self.data = mujoco.MjData(self.model)
        self.grid_samples = int(grid_samples)
        self.length_tol = float(length_tol)
        if self.grid_samples < 11:
            raise ValueError("grid_samples must be >= 11")

        self.qpos_addr = {name: self._qpos_addr(name) for name in (*PR_NAMES, *UL_NAMES)}
        self.joint_range = {name: self._joint_range(name) for name in (*PR_NAMES, *UL_NAMES)}
        self.tendon_targets = np.asarray(self.model.tendon_range[:4, 0], dtype=np.float64)

    @staticmethod
    def _load_model(xml_path: Path) -> mujoco.MjModel:
        try:
            return mujoco.MjModel.from_xml_path(str(xml_path))
        except ValueError as exc:
            fixed_xml = Lens110XmlTendonSolver._xml_with_resolved_meshdir(xml_path)
            if fixed_xml is None:
                raise
            try:
                return mujoco.MjModel.from_xml_string(fixed_xml)
            except ValueError:
                raise exc

    @staticmethod
    def _xml_with_resolved_meshdir(xml_path: Path) -> str | None:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        compiler = root.find("compiler")
        if compiler is None:
            return None
        meshdir = compiler.attrib.get("meshdir")
        if not meshdir:
            return None

        mesh_path = Path(meshdir)
        if mesh_path.is_absolute():
            return None
        current = (xml_path.parent / mesh_path).resolve()
        if current.exists():
            return None

        candidates = (
            (xml_path.parent.parent / "meshes").resolve(),
            (xml_path.parent / ".." / "meshes").resolve(),
        )
        for candidate in candidates:
            if candidate.exists():
                compiler.set("meshdir", str(candidate))
                return ET.tostring(root, encoding="unicode")
        return None

    def _joint_id(self, name: str) -> int:
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise KeyError(f"Joint not found in XML: {name}")
        return int(joint_id)

    def _qpos_addr(self, name: str) -> int:
        return int(self.model.jnt_qposadr[self._joint_id(name)])

    def _joint_range(self, name: str) -> tuple[float, float]:
        joint_id = self._joint_id(name)
        if self.model.jnt_limited[joint_id]:
            lo, hi = self.model.jnt_range[joint_id]
            return float(lo), float(hi)
        return -np.pi, np.pi

    def reset_qpos(self) -> None:
        self.data.qpos[:] = self.model.qpos0
        self.data.qvel[:] = 0.0

    def set_ankle_pose(
        self,
        *,
        left_pitch: float = 0.0,
        left_roll: float = 0.0,
        left_upper: float = 0.0,
        left_lower: float = 0.0,
        right_pitch: float = 0.0,
        right_roll: float = 0.0,
        right_upper: float = 0.0,
        right_lower: float = 0.0,
    ) -> None:
        values = {
            "left_ankle_pitch_joint": left_pitch,
            "left_ankle_roll_joint": left_roll,
            "left_ankle_upper_joint": left_upper,
            "left_ankle_lower_joint": left_lower,
            "right_ankle_pitch_joint": right_pitch,
            "right_ankle_roll_joint": right_roll,
            "right_ankle_upper_joint": right_upper,
            "right_ankle_lower_joint": right_lower,
        }
        for name, value in values.items():
            self.data.qpos[self.qpos_addr[name]] = float(value)

    def tendon_lengths(
        self,
        *,
        left_pitch: float = 0.0,
        left_roll: float = 0.0,
        left_upper: float = 0.0,
        left_lower: float = 0.0,
        right_pitch: float = 0.0,
        right_roll: float = 0.0,
        right_upper: float = 0.0,
        right_lower: float = 0.0,
    ) -> np.ndarray:
        self.reset_qpos()
        self.set_ankle_pose(
            left_pitch=left_pitch,
            left_roll=left_roll,
            left_upper=left_upper,
            left_lower=left_lower,
            right_pitch=right_pitch,
            right_roll=right_roll,
            right_upper=right_upper,
            right_lower=right_lower,
        )
        mujoco.mj_forward(self.model, self.data)
        return np.asarray(self.data.ten_length[:4], dtype=np.float64).copy()

    def _channel_error(self, channel: TendonChannel, pitch: float, roll: float, motor_angle: float) -> float:
        self.reset_qpos()
        self.data.qpos[self.qpos_addr[channel.pitch_joint]] = float(pitch)
        self.data.qpos[self.qpos_addr[channel.roll_joint]] = float(roll)
        self.data.qpos[self.qpos_addr[channel.solve_joint]] = float(motor_angle)
        mujoco.mj_forward(self.model, self.data)
        return float(self.data.ten_length[channel.tendon_id] - self.tendon_targets[channel.tendon_id])

    def _side_tendon_error(
        self,
        side: str,
        pitch: float,
        roll: float,
        upper: float,
        lower: float,
    ) -> np.ndarray:
        if side == "left":
            pitch_name, roll_name = "left_ankle_pitch_joint", "left_ankle_roll_joint"
            upper_name, lower_name = "left_ankle_upper_joint", "left_ankle_lower_joint"
            tendon_ids = np.array([0, 1], dtype=np.int32)
        elif side == "right":
            pitch_name, roll_name = "right_ankle_pitch_joint", "right_ankle_roll_joint"
            upper_name, lower_name = "right_ankle_upper_joint", "right_ankle_lower_joint"
            tendon_ids = np.array([2, 3], dtype=np.int32)
        else:
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")

        self.reset_qpos()
        self.data.qpos[self.qpos_addr[pitch_name]] = float(pitch)
        self.data.qpos[self.qpos_addr[roll_name]] = float(roll)
        self.data.qpos[self.qpos_addr[upper_name]] = float(upper)
        self.data.qpos[self.qpos_addr[lower_name]] = float(lower)
        mujoco.mj_forward(self.model, self.data)
        return np.asarray(self.data.ten_length[tendon_ids] - self.tendon_targets[tendon_ids], dtype=np.float64)

    def channel_length_derivative(
        self,
        channel: TendonChannel,
        pitch: float,
        roll: float,
        motor_angle: float,
        *,
        eps: float = 1e-5,
    ) -> float:
        lo, hi = self.joint_range[channel.solve_joint]
        plus = min(hi, motor_angle + eps)
        minus = max(lo, motor_angle - eps)
        if plus == minus:
            return 0.0
        f_plus = self._channel_error(channel, pitch, roll, plus)
        f_minus = self._channel_error(channel, pitch, roll, minus)
        return float((f_plus - f_minus) / (plus - minus))

    @staticmethod
    def _bisect(func, lo: float, hi: float, flo: float, fhi: float, tol: float, max_iter: int = 80) -> float:
        if abs(flo) <= tol:
            return lo
        if abs(fhi) <= tol:
            return hi
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            fmid = func(mid)
            if abs(fmid) <= tol or abs(hi - lo) <= 1e-12:
                return mid
            if flo * fmid <= 0.0:
                hi, fhi = mid, fmid
            else:
                lo, flo = mid, fmid
        return 0.5 * (lo + hi)

    @staticmethod
    def _golden_minimize(func, lo: float, hi: float, max_iter: int = 80) -> float:
        gr = (np.sqrt(5.0) - 1.0) / 2.0
        c = hi - gr * (hi - lo)
        d = lo + gr * (hi - lo)
        fc = func(c)
        fd = func(d)
        for _ in range(max_iter):
            if abs(hi - lo) <= 1e-12:
                break
            if fc < fd:
                hi, d, fd = d, c, fc
                c = hi - gr * (hi - lo)
                fc = func(c)
            else:
                lo, c, fc = c, d, fd
                d = lo + gr * (hi - lo)
                fd = func(d)
        return 0.5 * (lo + hi)

    def solve_channel(
        self,
        channel: TendonChannel,
        pitch: float,
        roll: float,
        *,
        initial: float | None = None,
    ) -> float:
        lo, hi = self.joint_range[channel.solve_joint]

        def error(angle: float) -> float:
            return self._channel_error(channel, pitch, roll, angle)

        if initial is not None and lo <= initial <= hi:
            center = float(initial)
            f_center = error(center)
            if abs(f_center) <= self.length_tol:
                return center
            for radius in (0.02, 0.05, 0.10, 0.20, 0.40, 0.80, 1.60, 3.20):
                a = max(lo, center - radius)
                b = min(hi, center + radius)
                fa = error(a)
                fb = error(b)
                if fa == 0.0 or fa * f_center < 0.0:
                    return self._bisect(error, a, center, fa, f_center, self.length_tol)
                if fb == 0.0 or f_center * fb < 0.0:
                    return self._bisect(error, center, b, f_center, fb, self.length_tol)
                if a <= lo and b >= hi:
                    break

        grid = np.linspace(lo, hi, self.grid_samples)
        if initial is not None and lo <= initial <= hi:
            grid = np.unique(np.sort(np.concatenate((grid, [float(initial)]))))

        values = np.asarray([error(x) for x in grid], dtype=np.float64)
        abs_values = np.abs(values)
        if abs_values.min() <= self.length_tol:
            near = np.flatnonzero(abs_values <= self.length_tol)
            if initial is None:
                return float(grid[near[np.argmin(np.abs(grid[near]))]])
            return float(grid[near[np.argmin(np.abs(grid[near] - initial))]])

        brackets: list[tuple[float, float, float, float]] = []
        for i in range(len(grid) - 1):
            f0, f1 = values[i], values[i + 1]
            if f0 == 0.0 or f0 * f1 < 0.0:
                brackets.append((float(grid[i]), float(grid[i + 1]), float(f0), float(f1)))

        if brackets:
            center = np.asarray([(a + b) * 0.5 for a, b, _, _ in brackets], dtype=np.float64)
            if initial is None:
                pick = int(np.argmin(np.abs(center)))
            else:
                pick = int(np.argmin(np.abs(center - initial)))
            a, b, fa, fb = brackets[pick]
            return self._bisect(error, a, b, fa, fb, self.length_tol)

        best = int(np.argmin(abs_values))
        a = float(grid[max(0, best - 1)])
        b = float(grid[min(len(grid) - 1, best + 1)])
        objective = lambda x: abs(error(x))
        angle = self._golden_minimize(objective, a, b)
        residual = abs(error(angle))
        if residual > max(1e-5, self.length_tol * 100.0):
            raise RuntimeError(
                f"No tendon-length root for {channel.side}_{channel.name}: "
                f"pitch={pitch:.6f}, roll={roll:.6f}, best_angle={angle:.6f}, "
                f"length_error={residual:.6g}"
            )
        return float(angle)

    def solve_side(
        self,
        side: str,
        pitch: float,
        roll: float,
        *,
        initial: Iterable[float] | None = None,
    ) -> np.ndarray:
        if side not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        init = None if initial is None else np.asarray(list(initial), dtype=np.float64)
        channels = [channel for channel in CHANNELS if channel.side == side]
        return np.asarray(
            [
                self.solve_channel(channel, pitch, roll, initial=None if init is None else init[i])
                for i, channel in enumerate(channels)
            ],
            dtype=np.float64,
        )

    def solve_pr_to_ul(
        self,
        left_pitch: float,
        left_roll: float,
        right_pitch: float,
        right_roll: float,
        *,
        initial: Iterable[float] | None = None,
    ) -> np.ndarray:
        init = None if initial is None else np.asarray(list(initial), dtype=np.float64)
        left_init = None if init is None else init[:2]
        right_init = None if init is None else init[2:4]
        left = self.solve_side("left", left_pitch, left_roll, initial=left_init)
        right = self.solve_side("right", right_pitch, right_roll, initial=right_init)
        return np.asarray([left[0], left[1], right[0], right[1]], dtype=np.float64)

    def _inverse_seed_points(self, side: str, initial: Iterable[float] | None = None) -> list[np.ndarray]:
        if side == "left":
            pitch_name, roll_name = "left_ankle_pitch_joint", "left_ankle_roll_joint"
        elif side == "right":
            pitch_name, roll_name = "right_ankle_pitch_joint", "right_ankle_roll_joint"
        else:
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")

        pitch_lo, pitch_hi = self.joint_range[pitch_name]
        roll_lo, roll_hi = self.joint_range[roll_name]
        qpos0 = np.array([self.model.qpos0[self.qpos_addr[pitch_name]], self.model.qpos0[self.qpos_addr[roll_name]]])
        seeds: list[np.ndarray] = []
        if initial is not None:
            seeds.append(np.asarray(list(initial), dtype=np.float64))
        seeds.extend(
            [
                qpos0,
                np.array([-0.15, 0.0], dtype=np.float64),
                np.array([-0.22, 0.0], dtype=np.float64),
                np.array([0.0, 0.0], dtype=np.float64),
                np.array([(pitch_lo + pitch_hi) * 0.5, (roll_lo + roll_hi) * 0.5], dtype=np.float64),
                np.array([pitch_lo, 0.0], dtype=np.float64),
                np.array([pitch_hi, 0.0], dtype=np.float64),
                np.array([-0.15, roll_lo], dtype=np.float64),
                np.array([-0.15, roll_hi], dtype=np.float64),
            ]
        )
        clipped: list[np.ndarray] = []
        for seed in seeds:
            value = np.asarray(seed, dtype=np.float64).reshape(2)
            value = np.array(
                [
                    np.clip(value[0], pitch_lo, pitch_hi),
                    np.clip(value[1], roll_lo, roll_hi),
                ],
                dtype=np.float64,
            )
            if not any(np.allclose(value, existing, atol=1e-12, rtol=0.0) for existing in clipped):
                clipped.append(value)
        return clipped

    def _solve_ul_side_custom(
        self,
        side: str,
        upper: float,
        lower: float,
        seed: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray],
    ) -> tuple[np.ndarray, float]:
        lo, hi = bounds

        def residual(x: np.ndarray) -> np.ndarray:
            return self._side_tendon_error(side, x[0], x[1], upper, lower)

        x = np.clip(np.asarray(seed, dtype=np.float64).reshape(2), lo, hi)
        best_x = x.copy()
        best_norm = float(np.linalg.norm(residual(x)))
        for _ in range(80):
            f0 = residual(x)
            norm0 = float(np.linalg.norm(f0))
            if norm0 < best_norm:
                best_x, best_norm = x.copy(), norm0
            if norm0 <= max(1e-8, self.length_tol):
                break

            jac = np.zeros((2, 2), dtype=np.float64)
            eps = 1e-5
            for j in range(2):
                xp = x.copy()
                xm = x.copy()
                xp[j] = min(hi[j], xp[j] + eps)
                xm[j] = max(lo[j], xm[j] - eps)
                if xp[j] == xm[j]:
                    continue
                jac[:, j] = (residual(xp) - residual(xm)) / (xp[j] - xm[j])

            lhs = jac.T @ jac + np.eye(2, dtype=np.float64) * 1e-8
            rhs = -(jac.T @ f0)
            try:
                step = np.linalg.solve(lhs, rhs)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

            accepted = False
            for alpha in (1.0, 0.5, 0.25, 0.1, 0.05, 0.01):
                candidate = np.clip(x + alpha * step, lo, hi)
                cand_norm = float(np.linalg.norm(residual(candidate)))
                if cand_norm < norm0:
                    x = candidate
                    accepted = True
                    break
            if not accepted:
                break

        final_norm = float(np.linalg.norm(residual(best_x)))
        return best_x, final_norm

    def solve_ul_side(
        self,
        side: str,
        upper: float,
        lower: float,
        *,
        initial: Iterable[float] | None = None,
    ) -> np.ndarray:
        if side == "left":
            pitch_name, roll_name = "left_ankle_pitch_joint", "left_ankle_roll_joint"
        elif side == "right":
            pitch_name, roll_name = "right_ankle_pitch_joint", "right_ankle_roll_joint"
        else:
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")

        lo = np.array([self.joint_range[pitch_name][0], self.joint_range[roll_name][0]], dtype=np.float64)
        hi = np.array([self.joint_range[pitch_name][1], self.joint_range[roll_name][1]], dtype=np.float64)
        seeds = self._inverse_seed_points(side, initial)

        best_x: np.ndarray | None = None
        best_norm = np.inf

        try:
            from scipy.optimize import least_squares
        except Exception:
            least_squares = None

        if least_squares is not None:
            for seed in seeds:
                result = least_squares(
                    lambda x: self._side_tendon_error(side, x[0], x[1], upper, lower),
                    seed,
                    bounds=(lo, hi),
                    xtol=1e-12,
                    ftol=1e-12,
                    gtol=1e-12,
                    max_nfev=100,
                )
                norm = float(np.linalg.norm(result.fun))
                if norm < best_norm:
                    best_x = np.asarray(result.x, dtype=np.float64)
                    best_norm = norm
                if best_norm <= max(1e-8, self.length_tol):
                    break

        if best_x is None or best_norm > max(1e-7, self.length_tol * 10.0):
            for seed in seeds:
                candidate, norm = self._solve_ul_side_custom(side, upper, lower, seed, (lo, hi))
                if norm < best_norm:
                    best_x = candidate
                    best_norm = norm
                if best_norm <= max(1e-8, self.length_tol):
                    break

        if best_x is None or best_norm > max(1e-5, self.length_tol * 100.0):
            raise RuntimeError(
                f"No pitch/roll root for {side}: upper={upper:.6f}, lower={lower:.6f}, "
                f"best_pitch_roll={None if best_x is None else best_x.tolist()}, length_error_norm={best_norm:.6g}"
            )
        return np.asarray(best_x, dtype=np.float64)

    def solve_ul_to_pr(
        self,
        left_upper: float,
        left_lower: float,
        right_upper: float,
        right_lower: float,
        *,
        initial: Iterable[float] | None = None,
    ) -> np.ndarray:
        init = None if initial is None else np.asarray(list(initial), dtype=np.float64)
        left_init = None if init is None else init[:2]
        right_init = None if init is None else init[2:4]
        left = self.solve_ul_side("left", left_upper, left_lower, initial=left_init)
        right = self.solve_ul_side("right", right_upper, right_lower, initial=right_init)
        return np.asarray([left[0], left[1], right[0], right[1]], dtype=np.float64)

    def solve_ul_to_pr_batch(
        self,
        ul_pos: np.ndarray,
        *,
        ul_vel: np.ndarray | None = None,
        dt: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        ul_pos = np.asarray(ul_pos, dtype=np.float64)
        if ul_pos.ndim != 2 or ul_pos.shape[1] != 4:
            raise ValueError("ul_pos must have shape (N, 4): left_upper,left_lower,right_upper,right_lower")

        pr_pos = np.zeros_like(ul_pos)
        last: np.ndarray | None = None
        for i, row in enumerate(ul_pos):
            pr_pos[i] = self.solve_ul_to_pr(row[0], row[1], row[2], row[3], initial=last)
            last = pr_pos[i]

        if ul_vel is None:
            if dt is None:
                return pr_pos, None
            return pr_pos, _finite_diff(pr_pos, dt)

        ul_vel = np.asarray(ul_vel, dtype=np.float64)
        if ul_vel.shape != ul_pos.shape:
            raise ValueError("ul_vel must have the same shape as ul_pos")
        pr_vel = np.zeros_like(ul_vel)
        for i, row in enumerate(pr_pos):
            left_j = self.jacobian_side("left", row[0], row[1])
            right_j = self.jacobian_side("right", row[2], row[3])
            pr_vel[i, :2] = np.linalg.lstsq(left_j, ul_vel[i, :2], rcond=None)[0]
            pr_vel[i, 2:4] = np.linalg.lstsq(right_j, ul_vel[i, 2:4], rcond=None)[0]
        return pr_pos, pr_vel

    def jacobian_side(self, side: str, pitch: float, roll: float, *, eps: float = 1e-5) -> np.ndarray:
        center = self.solve_side(side, pitch, roll)
        p_plus = self.solve_side(side, pitch + eps, roll, initial=center)
        p_minus = self.solve_side(side, pitch - eps, roll, initial=center)
        r_plus = self.solve_side(side, pitch, roll + eps, initial=center)
        r_minus = self.solve_side(side, pitch, roll - eps, initial=center)
        return np.column_stack(((p_plus - p_minus) / (2.0 * eps), (r_plus - r_minus) / (2.0 * eps)))

    def solve_pr_to_ul_batch(
        self,
        pr_pos: np.ndarray,
        *,
        pr_vel: np.ndarray | None = None,
        dt: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        pr_pos = np.asarray(pr_pos, dtype=np.float64)
        if pr_pos.ndim != 2 or pr_pos.shape[1] != 4:
            raise ValueError("pr_pos must have shape (N, 4): left_pitch,left_roll,right_pitch,right_roll")

        ul_pos = np.zeros_like(pr_pos)
        last: np.ndarray | None = None
        for i, row in enumerate(pr_pos):
            ul_pos[i] = self.solve_pr_to_ul(row[0], row[1], row[2], row[3], initial=last)
            last = ul_pos[i]

        if pr_vel is None:
            if dt is None:
                return ul_pos, None
            return ul_pos, _finite_diff(ul_pos, dt)

        pr_vel = np.asarray(pr_vel, dtype=np.float64)
        if pr_vel.shape != pr_pos.shape:
            raise ValueError("pr_vel must have the same shape as pr_pos")
        ul_vel = np.zeros_like(pr_vel)
        for i, row in enumerate(pr_pos):
            left_j = self.jacobian_side("left", row[0], row[1])
            right_j = self.jacobian_side("right", row[2], row[3])
            ul_vel[i, :2] = left_j @ pr_vel[i, :2]
            ul_vel[i, 2:4] = right_j @ pr_vel[i, 2:4]
        return ul_pos, ul_vel

    def check_solution(self, pr: Iterable[float], ul: Iterable[float]) -> np.ndarray:
        values = list(pr) + list(ul)
        if len(values) != 8:
            raise ValueError("expected 4 PR values followed by 4 UL values")
        lengths = self.tendon_lengths(
            left_pitch=values[0],
            left_roll=values[1],
            right_pitch=values[2],
            right_roll=values[3],
            left_upper=values[4],
            left_lower=values[5],
            right_upper=values[6],
            right_lower=values[7],
        )
        return lengths - self.tendon_targets

    def limit_margin(self, joint_name: str, value: float) -> tuple[float, float, float]:
        lo, hi = self.joint_range[joint_name]
        return float(value - lo), float(hi - value), float(min(value - lo, hi - value))

    def analyze_solution(self, pr: Iterable[float], ul: Iterable[float]) -> list[dict[str, float | str]]:
        pr_values = np.asarray(list(pr), dtype=np.float64)
        ul_values = np.asarray(list(ul), dtype=np.float64)
        if pr_values.shape != (4,) or ul_values.shape != (4,):
            raise ValueError("expected four PR values and four UL values")

        pitch_roll_by_side = {
            "left": (float(pr_values[0]), float(pr_values[1])),
            "right": (float(pr_values[2]), float(pr_values[3])),
        }
        rows: list[dict[str, float | str]] = []
        for channel, value in zip(CHANNELS, ul_values):
            pitch, roll = pitch_roll_by_side[channel.side]
            lo_margin, hi_margin, min_margin = self.limit_margin(channel.solve_joint, float(value))
            derivative = self.channel_length_derivative(channel, pitch, roll, float(value))
            rows.append(
                {
                    "joint": channel.solve_joint,
                    "angle": float(value),
                    "low_margin": lo_margin,
                    "high_margin": hi_margin,
                    "min_margin": min_margin,
                    "dlength_dmotor": derivative,
                    "inv_gain": float(np.inf if derivative == 0.0 else 1.0 / abs(derivative)),
                }
            )
        return rows


def _save_grid(args: argparse.Namespace) -> None:
    solver = Lens110XmlTendonSolver(args.xml, grid_samples=args.grid_samples, length_tol=args.length_tol)
    pitches = np.linspace(args.pitch_min, args.pitch_max, args.pitch_count)
    rolls = np.linspace(args.roll_min, args.roll_max, args.roll_count)
    rows = []
    last_left = None
    last_right = None
    for pitch in pitches:
        for roll in rolls:
            left = solver.solve_side("left", pitch, roll, initial=last_left)
            right = solver.solve_side("right", pitch, roll, initial=last_right)
            rows.append([pitch, roll, left[0], left[1], right[0], right[1]])
            last_left = left
            last_right = right
    out = np.asarray(rows, dtype=np.float64)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == ".npz":
        np.savez_compressed(
            output,
            pitch=out[:, 0],
            roll=out[:, 1],
            left_upper=out[:, 2],
            left_lower=out[:, 3],
            right_upper=out[:, 4],
            right_lower=out[:, 5],
            columns=np.asarray(["pitch", "roll", "left_upper", "left_lower", "right_upper", "right_lower"]),
        )
    else:
        header = "pitch,roll,left_upper,left_lower,right_upper,right_lower"
        np.savetxt(output, out, delimiter=",", header=header, comments="")
    print(f"[OK] wrote {len(out)} rows to {output}")


def _convert_npz_dir(args: argparse.Namespace) -> None:
    solver = Lens110XmlTendonSolver(args.xml, grid_samples=args.grid_samples, length_tol=args.length_tol)
    src_dir = Path(args.input_dir)
    dst_dir = Path(args.output_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(src_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No npz files found in {src_dir}")

    for path in paths:
        raw = np.load(path, allow_pickle=True)
        motion = {key: raw[key] for key in raw.files}
        dof_names = [str(name) for name in np.asarray(motion["dof_names"]).tolist()]
        dof_pos = np.asarray(motion["dof_pos"], dtype=np.float64)
        name_to_id = {name: i for i, name in enumerate(dof_names)}
        missing_pr = [name for name in PR_NAMES if name not in name_to_id]
        if missing_pr:
            raise KeyError(f"{path} missing PR joints: {missing_pr}")

        pr_pos = np.column_stack([dof_pos[:, name_to_id[name]] for name in PR_NAMES])
        pr_vel = None
        if "dof_vel" in motion:
            dof_vel = np.asarray(motion["dof_vel"], dtype=np.float64)
            pr_vel = np.column_stack([dof_vel[:, name_to_id[name]] for name in PR_NAMES])
        else:
            dof_vel = None

        fps = _as_scalar(motion.get("fps", 50.0))
        ul_pos, ul_vel = solver.solve_pr_to_ul_batch(pr_pos, pr_vel=pr_vel, dt=1.0 / fps)

        for j, name in enumerate(UL_NAMES):
            if name in name_to_id:
                dof_pos[:, name_to_id[name]] = ul_pos[:, j]
                if dof_vel is not None and ul_vel is not None:
                    dof_vel[:, name_to_id[name]] = ul_vel[:, j]
            else:
                name_to_id[name] = len(dof_names)
                dof_names.append(name)
                dof_pos = np.column_stack((dof_pos, ul_pos[:, j]))
                if dof_vel is not None:
                    value = ul_vel[:, j] if ul_vel is not None else _finite_diff(ul_pos[:, j : j + 1], 1.0 / fps)[:, 0]
                    dof_vel = np.column_stack((dof_vel, value))

        motion["dof_names"] = np.asarray(dof_names)
        motion["dof_pos"] = dof_pos.astype(np.float32) if args.float32 else dof_pos
        if dof_vel is not None:
            motion["dof_vel"] = dof_vel.astype(np.float32) if args.float32 else dof_vel
        motion["ankle_ul_source"] = np.asarray("lens110_xml_spatial_tendon_solve")
        motion["ankle_ul_order"] = np.asarray(UL_NAMES)

        out_path = dst_dir / path.name
        np.savez_compressed(out_path, **motion)
        print(f"[OK] {path.name} -> {out_path}")


def _cmd_solve(args: argparse.Namespace) -> None:
    solver = Lens110XmlTendonSolver(args.xml, grid_samples=args.grid_samples, length_tol=args.length_tol)
    ul = solver.solve_pr_to_ul(args.left_pitch, args.left_roll, args.right_pitch, args.right_roll)
    err = solver.check_solution((args.left_pitch, args.left_roll, args.right_pitch, args.right_roll), ul)
    print(f"left_upper  = {ul[0]:+.9f}")
    print(f"left_lower  = {ul[1]:+.9f}")
    print(f"right_upper = {ul[2]:+.9f}")
    print(f"right_lower = {ul[3]:+.9f}")
    print("tendon_error = " + np.array2string(err, precision=12, suppress_small=False))


def _cmd_solve_inverse(args: argparse.Namespace) -> None:
    solver = Lens110XmlTendonSolver(args.xml, grid_samples=args.grid_samples, length_tol=args.length_tol)
    pr = solver.solve_ul_to_pr(args.left_upper, args.left_lower, args.right_upper, args.right_lower)
    err = solver.check_solution(pr, (args.left_upper, args.left_lower, args.right_upper, args.right_lower))
    print(f"left_pitch  = {pr[0]:+.9f}")
    print(f"left_roll   = {pr[1]:+.9f}")
    print(f"right_pitch = {pr[2]:+.9f}")
    print(f"right_roll  = {pr[3]:+.9f}")
    print("tendon_error = " + np.array2string(err, precision=12, suppress_small=False))


def _cmd_check(args: argparse.Namespace) -> None:
    solver = Lens110XmlTendonSolver(args.xml, grid_samples=args.grid_samples, length_tol=args.length_tol)
    samples = (
        (-0.15, 0.0, -0.15, 0.0),
        (-0.2, 0.1, -0.2, -0.1),
        (-0.4, -0.15, -0.4, 0.15),
        (0.1, 0.05, 0.1, -0.05),
    )
    for pr in samples:
        ul = solver.solve_pr_to_ul(*pr)
        err = solver.check_solution(pr, ul)
        print(
            "PR=("
            + ", ".join(f"{x:+.6f}" for x in pr)
            + ") -> UL=("
            + ", ".join(f"{x:+.9f}" for x in ul)
            + f"), max_abs_tendon_error={np.max(np.abs(err)):.3e}"
        )


def _cmd_check_inverse(args: argparse.Namespace) -> None:
    solver = Lens110XmlTendonSolver(args.xml, grid_samples=args.grid_samples, length_tol=args.length_tol)
    samples = (
        (-0.15, 0.0, -0.15, 0.0),
        (-0.2, 0.1, -0.2, -0.1),
        (-0.4, -0.15, -0.4, 0.15),
        (0.1, 0.05, 0.1, -0.05),
    )
    last_pr = None
    last_ul = None
    for pr in samples:
        ul = solver.solve_pr_to_ul(*pr, initial=last_ul)
        recovered = solver.solve_ul_to_pr(*ul, initial=last_pr)
        err = solver.check_solution(recovered, ul)
        pr_error = recovered - np.asarray(pr, dtype=np.float64)
        print(
            "PR=("
            + ", ".join(f"{x:+.6f}" for x in pr)
            + ") -> UL=("
            + ", ".join(f"{x:+.9f}" for x in ul)
            + ") -> PR=("
            + ", ".join(f"{x:+.9f}" for x in recovered)
            + f"), max_abs_pr_error={np.max(np.abs(pr_error)):.3e}, "
            + f"max_abs_tendon_error={np.max(np.abs(err)):.3e}"
        )
        last_pr = recovered
        last_ul = ul


def _cmd_analyze(args: argparse.Namespace) -> None:
    solver = Lens110XmlTendonSolver(args.xml, grid_samples=args.grid_samples, length_tol=args.length_tol)
    pr = (args.left_pitch, args.left_roll, args.right_pitch, args.right_roll)
    ul = solver.solve_pr_to_ul(*pr)
    err = solver.check_solution(pr, ul)
    print("UL = " + np.array2string(ul, precision=9, suppress_small=False))
    print("tendon_error = " + np.array2string(err, precision=12, suppress_small=False))
    print("joint, angle, margin_to_low, margin_to_high, min_margin, dlength_dmotor, inv_gain")
    for row in solver.analyze_solution(pr, ul):
        print(
            f"{row['joint']}, "
            f"{row['angle']:+.9f}, "
            f"{row['low_margin']:+.6f}, "
            f"{row['high_margin']:+.6f}, "
            f"{row['min_margin']:+.6f}, "
            f"{row['dlength_dmotor']:+.9e}, "
            f"{row['inv_gain']:+.3f}"
        )


def _cmd_scan_singularity(args: argparse.Namespace) -> None:
    solver = Lens110XmlTendonSolver(args.xml, grid_samples=args.grid_samples, length_tol=args.length_tol)
    pitches = np.linspace(args.pitch_min, args.pitch_max, args.pitch_count)
    rolls = np.linspace(args.roll_min, args.roll_max, args.roll_count)
    worst_margin = (np.inf, None)
    worst_derivative = (np.inf, None)
    failures: list[str] = []
    last_ul: np.ndarray | None = None

    for pitch in pitches:
        for roll in rolls:
            pr = (float(pitch), float(roll), float(pitch), float(-roll))
            try:
                ul = solver.solve_pr_to_ul(*pr, initial=last_ul)
            except RuntimeError as exc:
                failures.append(f"pitch={pitch:+.6f}, roll={roll:+.6f}: {exc}")
                continue
            last_ul = ul
            for row in solver.analyze_solution(pr, ul):
                min_margin = float(row["min_margin"])
                abs_derivative = abs(float(row["dlength_dmotor"]))
                if min_margin < worst_margin[0]:
                    worst_margin = (min_margin, (pr, ul.copy(), row.copy()))
                if abs_derivative < worst_derivative[0]:
                    worst_derivative = (abs_derivative, (pr, ul.copy(), row.copy()))

    print(f"scanned {len(pitches) * len(rolls)} mirrored PR samples")
    print(f"failures = {len(failures)}")
    if failures:
        for item in failures[: args.max_failures]:
            print("[FAIL] " + item)
    if worst_margin[1] is not None:
        pr, ul, row = worst_margin[1]
        print(
            "[WORST_LIMIT] "
            f"pr={np.array2string(np.asarray(pr), precision=6)}, "
            f"ul={np.array2string(ul, precision=6)}, "
            f"joint={row['joint']}, min_margin={row['min_margin']:+.6f}"
        )
    if worst_derivative[1] is not None:
        pr, ul, row = worst_derivative[1]
        risk = "RISK" if worst_derivative[0] < args.derivative_threshold else "OK"
        print(
            "[WORST_SINGULAR] "
            f"{risk}, pr={np.array2string(np.asarray(pr), precision=6)}, "
            f"ul={np.array2string(ul, precision=6)}, "
            f"joint={row['joint']}, "
            f"|dlength_dmotor|={worst_derivative[0]:.9e}, "
            f"threshold={args.derivative_threshold:.3e}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", default=str(DEFAULT_XML), help="Lens110 MJCF XML path")
    parser.add_argument("--grid-samples", type=int, default=801)
    parser.add_argument("--length-tol", type=float, default=1e-8)
    sub = parser.add_subparsers(dest="command", required=True)

    solve = sub.add_parser("solve", help="solve one PR pose")
    solve.add_argument("--left-pitch", type=float, default=-0.15)
    solve.add_argument("--left-roll", type=float, default=0.0)
    solve.add_argument("--right-pitch", type=float, default=-0.15)
    solve.add_argument("--right-roll", type=float, default=0.0)
    solve.set_defaults(func=_cmd_solve)

    solve_inverse = sub.add_parser("solve-inverse", help="solve one UL pose back to PR")
    solve_inverse.add_argument("--left-upper", type=float, default=0.161341435)
    solve_inverse.add_argument("--left-lower", type=float, default=0.162273347)
    solve_inverse.add_argument("--right-upper", type=float, default=0.161341435)
    solve_inverse.add_argument("--right-lower", type=float, default=0.162273347)
    solve_inverse.set_defaults(func=_cmd_solve_inverse)

    check = sub.add_parser("check", help="run several tendon-length checks")
    check.set_defaults(func=_cmd_check)

    check_inverse = sub.add_parser("check-inverse", help="run PR->UL->PR round-trip checks")
    check_inverse.set_defaults(func=_cmd_check_inverse)

    analyze = sub.add_parser("analyze", help="solve one PR pose and report limit/singularity margins")
    analyze.add_argument("--left-pitch", type=float, default=-0.15)
    analyze.add_argument("--left-roll", type=float, default=0.0)
    analyze.add_argument("--right-pitch", type=float, default=-0.15)
    analyze.add_argument("--right-roll", type=float, default=0.0)
    analyze.set_defaults(func=_cmd_analyze)

    scan = sub.add_parser("scan-singularity", help="scan mirrored PR poses for joint-limit and singularity risk")
    scan.add_argument("--pitch-min", type=float, default=-1.047)
    scan.add_argument("--pitch-max", type=float, default=0.523)
    scan.add_argument("--pitch-count", type=int, default=61)
    scan.add_argument("--roll-min", type=float, default=-0.261)
    scan.add_argument("--roll-max", type=float, default=0.261)
    scan.add_argument("--roll-count", type=int, default=41)
    scan.add_argument("--derivative-threshold", type=float, default=1e-4)
    scan.add_argument("--max-failures", type=int, default=20)
    scan.set_defaults(func=_cmd_scan_singularity)

    grid = sub.add_parser("grid", help="write a PR->UL lookup grid")
    grid.add_argument("--pitch-min", type=float, default=-1.047)
    grid.add_argument("--pitch-max", type=float, default=0.523)
    grid.add_argument("--pitch-count", type=int, default=81)
    grid.add_argument("--roll-min", type=float, default=-0.261)
    grid.add_argument("--roll-max", type=float, default=0.261)
    grid.add_argument("--roll-count", type=int, default=61)
    grid.add_argument("--output", default=str(Path(__file__).resolve().parent / "lens110_xml_pr_to_ul_grid.npz"))
    grid.set_defaults(func=_save_grid)

    convert = sub.add_parser("convert-npz-dir", help="replace/add UL joints in MJLab npz files from PR joints")
    convert.add_argument("--input-dir", required=True)
    convert.add_argument("--output-dir", required=True)
    convert.add_argument("--float32", action="store_true", help="store dof_pos/dof_vel as float32")
    convert.set_defaults(func=_convert_npz_dir)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
