"""Evaluate PR/UL state models on pitchRoll2UpperLower/ankle_model data."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parent
PITCHROLL_DIR = ROOT.parent
ANKLE_MODEL_DIR = PITCHROLL_DIR / "ankle_model"
DEFAULT_OUTPUT = ROOT / "ankle_model_eval_vis"
DEFAULT_WEIGHTED_MLP_DIR = PITCHROLL_DIR / "t_p_v" / "models" / "weighted_mlp"
DEFAULT_POLY_DIR = PITCHROLL_DIR / "t_p_v" / "models" / "poly_degree3"
DEFAULT_OLD_PKL_DIR = ANKLE_MODEL_DIR / "models"
DEFAULT_GMR_PYTHON = Path(os.environ.get("LENS110_GMR_PYTHON", sys.executable))
DEFAULT_SKLEARN_PYTHON = Path(os.environ.get("LENS110_SKLEARN_PYTHON", sys.executable))

PR_COLS = (
    "left_ankle_pitch_joint_pos",
    "left_ankle_roll_joint_pos",
    "right_ankle_pitch_joint_pos",
    "right_ankle_roll_joint_pos",
    "left_ankle_pitch_joint_vel",
    "left_ankle_roll_joint_vel",
    "right_ankle_pitch_joint_vel",
    "right_ankle_roll_joint_vel",
)
UL_COLS = (
    "left_ankle_upper_joint_pos",
    "left_ankle_lower_joint_pos",
    "right_ankle_upper_joint_pos",
    "right_ankle_lower_joint_pos",
    "left_ankle_upper_joint_vel",
    "left_ankle_lower_joint_vel",
    "right_ankle_upper_joint_vel",
    "right_ankle_lower_joint_vel",
)
PR_OUT = ("left_pitch_pos", "left_roll_pos", "right_pitch_pos", "right_roll_pos", "left_pitch_vel", "left_roll_vel", "right_pitch_vel", "right_roll_vel")
UL_OUT = ("left_upper_pos", "left_lower_pos", "right_upper_pos", "right_lower_pos", "left_upper_vel", "left_lower_vel", "right_upper_vel", "right_lower_vel")


def clean_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    return env


def maybe_reexec() -> None:
    try:
        import torch  # noqa: F401
        import matplotlib  # noqa: F401
    except Exception:
        if os.environ.get("ANKLE_MODEL_EVAL_REEXECED") != "1" and DEFAULT_GMR_PYTHON.exists() and Path(sys.executable) != DEFAULT_GMR_PYTHON:
            env = clean_env()
            env["ANKLE_MODEL_EVAL_REEXECED"] = "1"
            os.execve(str(DEFAULT_GMR_PYTHON), [str(DEFAULT_GMR_PYTHON), "-B", str(Path(__file__).resolve()), *sys.argv[1:]], env)
        raise


def setup_matplotlib(output_dir: Path):
    mpl_dir = output_dir / ".mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def read_csv_matrix(path: Path, columns: tuple[str, ...]) -> np.ndarray:
    rows = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No columns in {path}")
        missing = [col for col in columns if col not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing columns in {path}: {missing}")
        for row in reader:
            values = [float(row[col]) for col in columns]
            if np.all(np.isfinite(values)):
                rows.append(values)
    return np.asarray(rows, dtype=np.float32)


def load_data(ankle_model_dir: Path, max_rows_per_file: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = []
    ys = []
    names = []
    for filename in ("left_joint_data_all.csv", "right_joint_data_all.csv"):
        path = ankle_model_dir / filename
        pr = read_csv_matrix(path, PR_COLS)
        ul = read_csv_matrix(path, UL_COLS)
        if max_rows_per_file is not None:
            pr = pr[:max_rows_per_file]
            ul = ul[:max_rows_per_file]
        xs.append(pr)
        ys.append(ul)
        names.extend([path.stem] * len(pr))
        print(f"[DATA] {path}: loaded {len(pr)} rows")
    return np.vstack(xs), np.vstack(ys), np.asarray(names, dtype=object)


def predict_weighted(model_dir: Path, relation: str, x: np.ndarray) -> np.ndarray:
    import torch

    model = torch.jit.load(str(model_dir / relation / "scripted.pth"), map_location="cpu").eval()
    with torch.no_grad():
        return model(torch.from_numpy(x.astype(np.float32))).cpu().numpy().astype(np.float32)


def predict_poly(model_dir: Path, relation: str, x: np.ndarray) -> np.ndarray:
    blob = np.load(model_dir / relation / "poly_relation.npz", allow_pickle=False)
    z = (x.astype(np.float64) - blob["input_mean"]) / blob["input_scale"]
    features = np.prod(np.power(z[:, None, :], blob["powers"][None, :, :]), axis=-1)
    return (features @ blob["coef"].T + blob["intercept"]).astype(np.float32)


def predict_old(model_dir: Path, relation: str, x: np.ndarray, output_dir: Path, sklearn_python: Path) -> np.ndarray:
    tmp = output_dir / "_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    inp = tmp / f"{relation}_input.npy"
    out = tmp / f"{relation}_output.npy"
    np.save(inp, x.astype(np.float64))
    code = """
import sys
from pathlib import Path
import joblib
import numpy as np

relation = sys.argv[1]
model_dir = Path(sys.argv[2])
x = np.load(sys.argv[3])
pos = x[:, 0:4]
vel = x[:, 4:8]
def load(name): return joblib.load(model_dir / name)
def opt(name):
    path = model_dir / name
    return joblib.load(path) if path.exists() else None
if relation == "pr_to_ul_state":
    m = opt("pos_pr2ul.pkl")
    pos_out = m.predict(pos) if m is not None else np.concatenate([load("pos_pr2ul_left.pkl").predict(pos[:,0:2]), load("pos_pr2ul_right.pkl").predict(pos[:,2:4])], axis=1)
    vel_out = load("vel_pr2ul.pkl").predict(vel)
else:
    m = opt("pos_ul2pr.pkl")
    pos_out = m.predict(pos) if m is not None else np.concatenate([load("pos_ul2pr_left.pkl").predict(pos[:,0:2]), load("pos_ul2pr_right.pkl").predict(pos[:,2:4])], axis=1)
    vel_out = load("vel_ul2pr.pkl").predict(vel)
np.save(sys.argv[4], np.concatenate([pos_out, vel_out], axis=1).astype(np.float32))
"""
    proc = subprocess.run([str(sklearn_python), "-B", "-c", code, relation, str(model_dir), str(inp), str(out)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=clean_env())
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    return np.load(out).astype(np.float32)


def metric_rows(relation: str, backend: str, y: np.ndarray, pred: np.ndarray, output_names: tuple[str, ...]) -> list[dict]:
    err = pred - y
    rows = []
    rows.append({"relation": relation, "backend": backend, "source": "ankle_model", "output": "__all__", "count": len(y), "mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err * err))), "max_abs": float(np.max(np.abs(err))), "bias": float(np.mean(err))})
    for i, name in enumerate(output_names):
        e = err[:, i]
        rows.append({"relation": relation, "backend": backend, "source": "ankle_model", "output": name, "count": len(y), "mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e * e))), "max_abs": float(np.max(np.abs(e))), "bias": float(np.mean(e))})
    return rows


def plot_images(plt, relation_dir: Path, backend: str, y: np.ndarray, pred: np.ndarray, output_names: tuple[str, ...]) -> None:
    backend_dir = relation_dir / backend
    backend_dir.mkdir(parents=True, exist_ok=True)
    n = len(y)
    idx = np.linspace(0, n - 1, min(n, 6000)).astype(np.int64)
    err = pred - y
    cols = 4
    rows = int(np.ceil(len(output_names) / cols))
    for kind in ("scatter_true_vs_pred", "error_hist", "timeseries"):
        fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.2 * rows), squeeze=False)
        for j, name in enumerate(output_names):
            ax = axes[j // cols][j % cols]
            if kind == "scatter_true_vs_pred":
                ax.scatter(y[idx, j], pred[idx, j], s=3, alpha=0.35)
                lo = min(float(y[idx, j].min()), float(pred[idx, j].min()))
                hi = max(float(y[idx, j].max()), float(pred[idx, j].max()))
                ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)
                ax.set_xlabel("true")
                ax.set_ylabel("pred")
            elif kind == "error_hist":
                ax.hist(err[:, j], bins=80)
                ax.set_xlabel("pred - true")
            else:
                t = np.arange(min(900, n))
                ax.plot(t, y[: len(t), j], label="true", linewidth=1)
                ax.plot(t, pred[: len(t), j], label="pred", linewidth=1)
                if j == 0:
                    ax.legend()
            ax.set_title(name)
            ax.grid(alpha=0.2)
        for k in range(len(output_names), rows * cols):
            axes[k // cols][k % cols].axis("off")
        fig.tight_layout()
        fig.savefig(backend_dir / f"{kind}.png", dpi=160)
        plt.close(fig)


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = ["relation", "backend", "source", "output", "count", "mae", "rmse", "max_abs", "bias"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_overall_plot(plt, output_dir: Path, relation: str, rows: list[dict]) -> None:
    relation_dir = output_dir / relation
    relation_dir.mkdir(parents=True, exist_ok=True)
    overall = [r for r in rows if r["relation"] == relation and r["output"] == "__all__"]
    labels = [r["backend"] for r in overall]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.18, [r["mae"] for r in overall], width=0.36, label="MAE")
    ax.bar(x + 0.18, [r["rmse"] for r in overall], width=0.36, label="RMSE")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("raw unit error")
    ax.set_title(f"ankle_model / {relation}")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(relation_dir / "overall_mae_rmse.png", dpi=160)
    plt.close(fig)


def evaluate(args: argparse.Namespace) -> None:
    maybe_reexec()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt = setup_matplotlib(output_dir)
    pr, ul, _source = load_data(Path(args.ankle_model_dir), args.max_rows_per_file)
    jobs = [
        ("pr_to_ul_state", pr, ul, UL_OUT),
        ("ul_to_pr_state", ul, pr, PR_OUT),
    ]
    all_rows: list[dict] = []
    for relation, x, y, outputs in jobs:
        relation_rows: list[dict] = []
        for backend in ("weighted_mlp", "poly_degree3", "old_pkl"):
            if backend == "weighted_mlp":
                pred = predict_weighted(Path(args.weighted_mlp_dir), relation, x)
            elif backend == "poly_degree3":
                pred = predict_poly(Path(args.poly_dir), relation, x)
            else:
                pred = predict_old(Path(args.old_pkl_dir), relation, x, output_dir, Path(args.sklearn_python))
            rows = metric_rows(relation, backend, y, pred, outputs)
            all_rows.extend(rows)
            relation_rows.extend(rows)
            plot_images(plt, output_dir / relation, backend, y, pred, outputs)
            overall = rows[0]
            print(f"[OK] {relation} {backend}: MAE={overall['mae']:.6g} RMSE={overall['rmse']:.6g} MaxAbs={overall['max_abs']:.6g}")
        write_overall_plot(plt, output_dir, relation, relation_rows)
    write_csv(output_dir / "summary.csv", all_rows)
    (output_dir / "summary.json").write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    print(f"[WROTE] {output_dir / 'summary.csv'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate models on ankle_model CSV data.")
    parser.add_argument("--ankle_model_dir", default=str(ANKLE_MODEL_DIR))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--weighted_mlp_dir", default=str(DEFAULT_WEIGHTED_MLP_DIR))
    parser.add_argument("--poly_dir", default=str(DEFAULT_POLY_DIR))
    parser.add_argument("--old_pkl_dir", default=str(DEFAULT_OLD_PKL_DIR))
    parser.add_argument("--sklearn_python", default=str(DEFAULT_SKLEARN_PYTHON))
    parser.add_argument("--max_rows_per_file", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
