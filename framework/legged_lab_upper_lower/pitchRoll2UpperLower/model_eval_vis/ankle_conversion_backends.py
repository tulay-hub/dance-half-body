from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parent
PITCHROLL_DIR = ROOT.parent
TPV_DIR = PITCHROLL_DIR / "t_p_v"
DEFAULT_WEIGHTED_MLP_DIR = TPV_DIR / "models" / "weighted_mlp"
DEFAULT_POLY_DEGREE3_DIR = TPV_DIR / "models" / "poly_degree3"
DEFAULT_OLD_PKL_DIR = PITCHROLL_DIR / "ankle_model" / "models"
DEFAULT_TORCH_PYTHON = Path(os.environ.get("LENS110_TORCH_PYTHON", sys.executable))
DEFAULT_SKLEARN_PYTHON = Path(os.environ.get("LENS110_SKLEARN_PYTHON", sys.executable))

PR_POS = ("left_pitch_pos", "left_roll_pos", "right_pitch_pos", "right_roll_pos")
PR_VEL = ("left_pitch_vel", "left_roll_vel", "right_pitch_vel", "right_roll_vel")
UL_POS = ("left_upper_pos", "left_lower_pos", "right_upper_pos", "right_lower_pos")
UL_VEL = ("left_upper_vel", "left_lower_vel", "right_upper_vel", "right_lower_vel")


class WeightedMlpBackend:
    label = "weighted_mlp"

    def __init__(
        self,
        model_dir: str | Path = DEFAULT_WEIGHTED_MLP_DIR,
        external_python: str | Path | None = DEFAULT_TORCH_PYTHON,
    ):
        self.model_dir = Path(model_dir)
        self.external_python = Path(external_python) if external_python else None
        self.external_reason = None
        try:
            import torch

            self.pr_to_ul = torch.jit.load(
                str(self.model_dir / "pr_to_ul_state" / "scripted.pth"), map_location="cpu"
            ).eval()
            self.ul_to_pr = torch.jit.load(
                str(self.model_dir / "ul_to_pr_state" / "scripted.pth"), map_location="cpu"
            ).eval()
        except Exception as exc:
            if _external_allowed(self.external_python):
                self.external_reason = str(exc)
                self.pr_to_ul = None
                self.ul_to_pr = None
            else:
                raise

    def convert(self, mode: str, values: list[float]) -> tuple[tuple[str, ...], list[float]]:
        if self.external_reason is not None:
            return _external_convert("weighted_mlp", self.model_dir, self.external_python, mode, values)
        import torch

        x = _as_state_tensor(values)
        with torch.no_grad():
            if mode == "pr_to_ul":
                y = self.pr_to_ul(x)
                return UL_POS + UL_VEL, y[0].tolist()
            if mode == "ul_to_pr":
                y = self.ul_to_pr(x)
                return PR_POS + PR_VEL, y[0].tolist()
        raise ValueError(f"Unsupported mode: {mode}")


class PolyDegree3Backend:
    label = "poly_degree3"

    def __init__(self, model_dir: str | Path = DEFAULT_POLY_DEGREE3_DIR):
        self.model_dir = Path(model_dir)
        self.pr_to_ul = np.load(self.model_dir / "pr_to_ul_state" / "poly_relation.npz", allow_pickle=False)
        self.ul_to_pr = np.load(self.model_dir / "ul_to_pr_state" / "poly_relation.npz", allow_pickle=False)

    def convert(self, mode: str, values: list[float]) -> tuple[tuple[str, ...], list[float]]:
        x = _as_state_array(values)
        if mode == "pr_to_ul":
            y = self._predict(self.pr_to_ul, x)
            return UL_POS + UL_VEL, y[0].astype(float).tolist()
        if mode == "ul_to_pr":
            y = self._predict(self.ul_to_pr, x)
            return PR_POS + PR_VEL, y[0].astype(float).tolist()
        raise ValueError(f"Unsupported mode: {mode}")

    @staticmethod
    def _predict(blob, x: np.ndarray) -> np.ndarray:
        z = (x - blob["input_mean"]) / blob["input_scale"]
        powers = blob["powers"]
        features = np.prod(np.power(z[:, None, :], powers[None, :, :]), axis=-1)
        return features @ blob["coef"].T + blob["intercept"]


class OldPklBackend:
    label = "old_pkl"

    def __init__(
        self,
        model_dir: str | Path = DEFAULT_OLD_PKL_DIR,
        external_python: str | Path | None = DEFAULT_SKLEARN_PYTHON,
    ):
        self.model_dir = Path(model_dir)
        self.external_python = Path(external_python) if external_python else None
        self.external_reason = None
        try:
            self.pos_pr2ul = self._load_optional("pos_pr2ul.pkl")
            self.pos_ul2pr = self._load_optional("pos_ul2pr.pkl")
            self.vel_pr2ul = self._load_optional("vel_pr2ul.pkl")
            self.vel_ul2pr = self._load_optional("vel_ul2pr.pkl")
            self.pos_pr2ul_left = self._load_optional("pos_pr2ul_left.pkl")
            self.pos_pr2ul_right = self._load_optional("pos_pr2ul_right.pkl")
            self.pos_ul2pr_left = self._load_optional("pos_ul2pr_left.pkl")
            self.pos_ul2pr_right = self._load_optional("pos_ul2pr_right.pkl")
        except Exception as exc:
            if _external_allowed(self.external_python):
                self.external_reason = str(exc)
            else:
                raise

    def convert(self, mode: str, values: list[float]) -> tuple[tuple[str, ...], list[float]]:
        if self.external_reason is not None:
            return _external_convert("old_pkl", self.model_dir, self.external_python, mode, values)
        x = _as_state_array(values)
        pos = x[:, 0:4]
        vel = x[:, 4:8]
        if mode == "pr_to_ul":
            pos_out = self._predict_pos_pr_to_ul(pos)
            vel_out = self._predict_required(self.vel_pr2ul, vel, "vel_pr2ul.pkl")
            return UL_POS + UL_VEL, np.concatenate([pos_out, vel_out], axis=-1)[0].astype(float).tolist()
        if mode == "ul_to_pr":
            pos_out = self._predict_pos_ul_to_pr(pos)
            vel_out = self._predict_required(self.vel_ul2pr, vel, "vel_ul2pr.pkl")
            return PR_POS + PR_VEL, np.concatenate([pos_out, vel_out], axis=-1)[0].astype(float).tolist()
        raise ValueError(f"Unsupported mode: {mode}")

    def _load_optional(self, filename: str):
        import joblib

        path = self.model_dir / filename
        return joblib.load(path) if path.exists() else None

    @staticmethod
    def _predict_required(model, x: np.ndarray, name: str) -> np.ndarray:
        if model is None:
            raise FileNotFoundError(f"Missing required old pkl model: {name}")
        return model.predict(x)

    def _predict_pos_pr_to_ul(self, pos: np.ndarray) -> np.ndarray:
        if self.pos_pr2ul is not None:
            return self.pos_pr2ul.predict(pos)
        left = self._predict_required(self.pos_pr2ul_left, pos[:, 0:2], "pos_pr2ul_left.pkl")
        right = self._predict_required(self.pos_pr2ul_right, pos[:, 2:4], "pos_pr2ul_right.pkl")
        return np.concatenate([left, right], axis=-1)

    def _predict_pos_ul_to_pr(self, pos: np.ndarray) -> np.ndarray:
        if self.pos_ul2pr is not None:
            return self.pos_ul2pr.predict(pos)
        left = self._predict_required(self.pos_ul2pr_left, pos[:, 0:2], "pos_ul2pr_left.pkl")
        right = self._predict_required(self.pos_ul2pr_right, pos[:, 2:4], "pos_ul2pr_right.pkl")
        return np.concatenate([left, right], axis=-1)


class ConversionBackends:
    def __init__(
        self,
        weighted_mlp_dir: str | Path = DEFAULT_WEIGHTED_MLP_DIR,
        poly_degree3_dir: str | Path = DEFAULT_POLY_DEGREE3_DIR,
        old_pkl_dir: str | Path = DEFAULT_OLD_PKL_DIR,
        weighted_mlp_python: str | Path | None = DEFAULT_TORCH_PYTHON,
        old_pkl_python: str | Path | None = DEFAULT_SKLEARN_PYTHON,
    ):
        self.backend_specs = {
            "weighted_mlp": (WeightedMlpBackend, weighted_mlp_dir, weighted_mlp_python),
            "poly_degree3": (PolyDegree3Backend, poly_degree3_dir),
            "old_pkl": (OldPklBackend, old_pkl_dir, old_pkl_python),
        }
        self.backends = {}

    def labels(self) -> list[str]:
        return list(self.backend_specs)

    def describe(self, backend: str) -> str:
        selected = self._get(backend)
        return f"{backend}: {selected.model_dir}"

    def convert(self, backend: str, mode: str, values: list[float]) -> tuple[tuple[str, ...], list[float]]:
        return self._get(backend).convert(mode, values)

    def _get(self, backend: str):
        if backend not in self.backend_specs:
            raise ValueError(f"Unsupported backend: {backend}")
        if backend not in self.backends:
            spec = self.backend_specs[backend]
            cls, args = spec[0], spec[1:]
            self.backends[backend] = cls(*args)
        return self.backends[backend]


def _as_state_tensor(values: list[float]):
    import torch

    if len(values) != 8:
        raise ValueError("Expected 8 values: 4 positions + 4 velocities.")
    return torch.tensor([values], dtype=torch.float32)


def _as_state_array(values: list[float]) -> np.ndarray:
    if len(values) != 8:
        raise ValueError("Expected 8 values: 4 positions + 4 velocities.")
    return np.asarray([values], dtype=np.float64)


def _external_allowed(python: Path | None) -> bool:
    return (
        python is not None
        and python.exists()
        and python != Path(sys.executable)
        and os.environ.get("ANKLE_CONVERSION_NO_EXTERNAL") != "1"
    )


def _external_convert(
    backend: str,
    model_dir: Path,
    python: Path | None,
    mode: str,
    values: list[float],
) -> tuple[tuple[str, ...], list[float]]:
    if python is None:
        raise RuntimeError(f"No external Python configured for backend {backend}.")
    payload = {"backend": backend, "model_dir": str(model_dir), "mode": mode, "values": values}
    code = (
        "import json, sys; "
        f"sys.path.insert(0, {str(ROOT)!r}); "
        "from ankle_conversion_backends import WeightedMlpBackend, OldPklBackend; "
        "p=json.loads(sys.stdin.read()); "
        "cls=WeightedMlpBackend if p['backend']=='weighted_mlp' else OldPklBackend; "
        "m=cls(p['model_dir'], external_python=None); "
        "names, values=m.convert(p['mode'], p['values']); "
        "print(json.dumps({'names': names, 'values': values}))"
    )
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["ANKLE_CONVERSION_NO_EXTERNAL"] = "1"
    env["PYTHONPATH"] = f"{ROOT}:{env.get('PYTHONPATH', '')}"
    proc = subprocess.run(
        [str(python), "-B", "-c", code],
        input=json.dumps(payload),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"External backend {backend} failed with {python}: {proc.stderr.strip() or proc.stdout.strip()}"
        )
    result = json.loads(proc.stdout)
    return tuple(result["names"]), result["values"]
