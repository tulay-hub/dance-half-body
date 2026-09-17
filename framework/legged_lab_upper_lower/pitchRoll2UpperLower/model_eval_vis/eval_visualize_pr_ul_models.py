"""Evaluate and visualize PR/UL ankle relation models.

This compares:
    - weighted_mlp: t_p_v/models/weighted_mlp
    - poly_degree3: t_p_v/models/poly_degree3
    - old_pkl: pitchRoll2UpperLower/ankle_model/models, state relations only

Run:
    python -B pitchRoll2UpperLower/model_eval_vis/eval_visualize_pr_ul_models.py
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PITCHROLL_DIR = SCRIPT_DIR.parent
REPO_ROOT = PITCHROLL_DIR.parent
TPV_DIR = PITCHROLL_DIR / "t_p_v"
DEFAULT_DATA = TPV_DIR / "data" / "test"
DEFAULT_OUTPUT = SCRIPT_DIR / "csv_eval_vis"
DEFAULT_WEIGHTED_MLP_DIR = TPV_DIR / "models" / "weighted_mlp"
DEFAULT_POLY_DIR = TPV_DIR / "models" / "poly_degree3"
DEFAULT_OLD_PKL_DIR = PITCHROLL_DIR / "ankle_model" / "models"
DEFAULT_GMR_PYTHON = Path(os.environ.get("LENS110_GMR_PYTHON", sys.executable))
DEFAULT_SKLEARN_PYTHON = Path(os.environ.get("LENS110_SKLEARN_PYTHON", sys.executable))

PR_POS = ("left_pitch_pos", "left_roll_pos", "right_pitch_pos", "right_roll_pos")
PR_VEL = ("left_pitch_vel", "left_roll_vel", "right_pitch_vel", "right_roll_vel")
PR_TAU = ("left_pitch_tau", "left_roll_tau", "right_pitch_tau", "right_roll_tau")
UL_POS = ("left_upper_pos", "left_lower_pos", "right_upper_pos", "right_lower_pos")
UL_VEL = ("left_upper_vel", "left_lower_vel", "right_upper_vel", "right_lower_vel")
UL_TAU = ("left_upper_tau", "left_lower_tau", "right_upper_tau", "right_lower_tau")

RELATION_COLUMNS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "pr_to_ul_state": (PR_POS + PR_VEL, UL_POS + UL_VEL),
    "ul_to_pr_state": (UL_POS + UL_VEL, PR_POS + PR_VEL),
    "pr_to_ul_torque": (PR_POS + PR_VEL + PR_TAU, UL_TAU),
    "ul_to_pr_torque": (UL_POS + UL_VEL + UL_TAU, PR_TAU),
    "pr_to_ul_full": (PR_POS + PR_VEL + PR_TAU, UL_POS + UL_VEL + UL_TAU),
    "ul_to_pr_full": (UL_POS + UL_VEL + UL_TAU, PR_POS + PR_VEL + PR_TAU),
}
BACKENDS = ("weighted_mlp", "poly_degree3", "old_pkl")
OLD_PKL_RELATIONS = {"pr_to_ul_state", "ul_to_pr_state"}


def clean_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    if extra:
        env.update(extra)
    return env


def maybe_reexec_for_plotting() -> None:
    missing = [
        name
        for name in ("torch", "matplotlib")
        if importlib.util.find_spec(name) is None
    ]
    if (
        missing
        and os.environ.get("ANKLE_EVAL_REEXECED") != "1"
        and DEFAULT_GMR_PYTHON.exists()
        and Path(sys.executable) != DEFAULT_GMR_PYTHON
    ):
        env = clean_env({"ANKLE_EVAL_REEXECED": "1"})
        print(f"[INFO] missing {missing}; relaunching with {DEFAULT_GMR_PYTHON}")
        os.execve(
            str(DEFAULT_GMR_PYTHON),
            [str(DEFAULT_GMR_PYTHON), "-B", str(Path(__file__).resolve()), *sys.argv[1:]],
            env,
        )


def resolve_csv_paths(value: str) -> list[Path]:
    out: list[Path] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        path = Path(item)
        if path.is_dir():
            out.extend(sorted(path.glob("*.csv")))
        else:
            out.append(path)
    missing = [str(path) for path in out if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing CSV files: " + ", ".join(missing))
    if not out:
        raise ValueError("No CSV files found.")
    return out


def infer_source(path: Path, row: dict[str, str]) -> str:
    raw = (row.get("source") or path.stem).lower()
    if "policy" in raw:
        return "policy"
    if "suspended" in raw or "base-held" in raw or "base_held" in raw:
        return "suspended"
    return "other"


def parse_float(row: dict[str, str], column: str) -> float:
    value = row[column]
    if value == "":
        return math.nan
    return float(value)


def row_is_valid(values: Iterable[float], max_abs_tau: float | None, max_abs_vel: float | None, columns: list[str]) -> bool:
    vals = list(values)
    if not np.all(np.isfinite(vals)):
        return False
    if max_abs_tau is not None:
        for col, value in zip(columns, vals):
            if col.endswith("_tau") and abs(value) > max_abs_tau:
                return False
    if max_abs_vel is not None:
        for col, value in zip(columns, vals):
            if col.endswith("_vel") and abs(value) > max_abs_vel:
                return False
    return True


def load_relation_data(
    csv_paths: list[Path],
    input_columns: tuple[str, ...],
    output_columns: tuple[str, ...],
    max_abs_tau: float | None,
    max_abs_vel: float | None,
    max_rows_per_file: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs: list[list[float]] = []
    ys: list[list[float]] = []
    sources: list[str] = []
    file_ids: list[int] = []
    needed = list(dict.fromkeys(input_columns + output_columns))
    for file_id, path in enumerate(csv_paths):
        loaded = 0
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                continue
            missing = [col for col in needed if col not in reader.fieldnames]
            if missing:
                print(f"[SKIP] {path}: missing columns {missing}")
                continue
            for row in reader:
                x = [parse_float(row, col) for col in input_columns]
                y = [parse_float(row, col) for col in output_columns]
                all_values = x + y
                if not row_is_valid(all_values, max_abs_tau, max_abs_vel, list(input_columns + output_columns)):
                    continue
                xs.append(x)
                ys.append(y)
                sources.append(infer_source(path, row))
                file_ids.append(file_id)
                loaded += 1
                if max_rows_per_file is not None and loaded >= max_rows_per_file:
                    break
        print(f"[DATA] {path}: loaded {loaded} rows")
    if not xs:
        raise ValueError("No usable rows for relation.")
    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(ys, dtype=np.float32),
        np.asarray(sources, dtype=object),
        np.asarray(file_ids, dtype=np.int32),
    )


def predict_weighted(model_dir: Path, relation: str, x: np.ndarray, batch_size: int) -> np.ndarray:
    import torch

    model = torch.jit.load(str(model_dir / relation / "scripted.pth"), map_location="cpu").eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start : start + batch_size].astype(np.float32))
            preds.append(model(xb).cpu().numpy())
    return np.vstack(preds).astype(np.float32)


def predict_poly(model_dir: Path, relation: str, x: np.ndarray) -> np.ndarray:
    blob = np.load(model_dir / relation / "poly_relation.npz", allow_pickle=False)
    z = (x.astype(np.float64) - blob["input_mean"]) / blob["input_scale"]
    powers = blob["powers"]
    features = np.prod(np.power(z[:, None, :], powers[None, :, :]), axis=-1)
    y = features @ blob["coef"].T + blob["intercept"]
    return y.astype(np.float32)


def predict_old_pkl_external(
    model_dir: Path,
    relation: str,
    x: np.ndarray,
    tmp_dir: Path,
    sklearn_python: Path,
) -> np.ndarray:
    if relation not in OLD_PKL_RELATIONS:
        raise ValueError(f"old_pkl does not support {relation}")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    input_path = tmp_dir / f"old_pkl_{relation}_input.npy"
    output_path = tmp_dir / f"old_pkl_{relation}_output.npy"
    np.save(input_path, x.astype(np.float64))
    code = """
import sys
from pathlib import Path

import joblib
import numpy as np

relation = sys.argv[1]
model_dir = Path(sys.argv[2])
inp = Path(sys.argv[3])
out = Path(sys.argv[4])
x = np.load(inp)
pos = x[:, 0:4]
vel = x[:, 4:8]


def load(name):
    return joblib.load(model_dir / name)


def load_optional(name):
    path = model_dir / name
    return joblib.load(path) if path.exists() else None


if relation == "pr_to_ul_state":
    pos_model = load_optional("pos_pr2ul.pkl")
    if pos_model is not None:
        pos_out = pos_model.predict(pos)
    else:
        pos_out = np.concatenate(
            [
                load("pos_pr2ul_left.pkl").predict(pos[:, 0:2]),
                load("pos_pr2ul_right.pkl").predict(pos[:, 2:4]),
            ],
            axis=1,
        )
    vel_out = load("vel_pr2ul.pkl").predict(vel)
elif relation == "ul_to_pr_state":
    pos_model = load_optional("pos_ul2pr.pkl")
    if pos_model is not None:
        pos_out = pos_model.predict(pos)
    else:
        pos_out = np.concatenate(
            [
                load("pos_ul2pr_left.pkl").predict(pos[:, 0:2]),
                load("pos_ul2pr_right.pkl").predict(pos[:, 2:4]),
            ],
            axis=1,
        )
    vel_out = load("vel_ul2pr.pkl").predict(vel)
else:
    raise SystemExit("unsupported relation " + relation)

np.save(out, np.concatenate([pos_out, vel_out], axis=1).astype(np.float32))
"""
    proc = subprocess.run(
        [str(sklearn_python), "-B", "-c", code, relation, str(model_dir), str(input_path), str(output_path)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=clean_env(),
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    return np.load(output_path).astype(np.float32)


def compute_metrics(y: np.ndarray, pred: np.ndarray, output_columns: tuple[str, ...], source: np.ndarray) -> list[dict]:
    err = pred - y
    rows: list[dict] = []
    masks = {"all": np.ones(len(y), dtype=bool)}
    for name in ("policy", "suspended", "other"):
        masks[name] = source == name
    for source_name, mask in masks.items():
        if not np.any(mask):
            continue
        e = err[mask]
        rows.append(
            {
                "source": source_name,
                "output": "__all__",
                "count": int(mask.sum()),
                "mae": float(np.mean(np.abs(e))),
                "rmse": float(np.sqrt(np.mean(e * e))),
                "max_abs": float(np.max(np.abs(e))),
                "bias": float(np.mean(e)),
            }
        )
        for j, col in enumerate(output_columns):
            ej = e[:, j]
            rows.append(
                {
                    "source": source_name,
                    "output": col,
                    "count": int(mask.sum()),
                    "mae": float(np.mean(np.abs(ej))),
                    "rmse": float(np.sqrt(np.mean(ej * ej))),
                    "max_abs": float(np.max(np.abs(ej))),
                    "bias": float(np.mean(ej)),
                }
            )
    return rows


def safe_name(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_").replace(":", "_")


def setup_matplotlib(output_dir: Path):
    mpl_config = output_dir / ".mplconfig"
    mpl_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_backend_summary(plt, relation_dir: Path, relation: str, summary_rows: list[dict]) -> None:
    rows = [r for r in summary_rows if r["source"] == "all" and r["output"] == "__all__"]
    if not rows:
        return
    labels = [r["backend"] for r in rows]
    mae = [r["mae"] for r in rows]
    rmse = [r["rmse"] for r in rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(x - 0.18, mae, width=0.36, label="MAE")
    ax.bar(x + 0.18, rmse, width=0.36, label="RMSE")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(f"{relation} overall error")
    ax.set_ylabel("raw unit error")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(relation_dir / "overall_mae_rmse.png", dpi=160)
    plt.close(fig)


def plot_prediction_details(
    plt,
    relation_dir: Path,
    relation: str,
    backend: str,
    output_columns: tuple[str, ...],
    y: np.ndarray,
    pred: np.ndarray,
    max_points: int,
    time_series_points: int,
) -> None:
    backend_dir = relation_dir / backend
    backend_dir.mkdir(parents=True, exist_ok=True)
    n = len(y)
    if n > max_points:
        idx = np.linspace(0, n - 1, max_points).astype(np.int64)
    else:
        idx = np.arange(n)
    err = pred - y

    cols = min(4, len(output_columns))
    rows = int(math.ceil(len(output_columns) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.6 * rows), squeeze=False)
    for j, col in enumerate(output_columns):
        ax = axes[j // cols][j % cols]
        ax.scatter(y[idx, j], pred[idx, j], s=4, alpha=0.35)
        lo = float(min(y[idx, j].min(), pred[idx, j].min()))
        hi = float(max(y[idx, j].max(), pred[idx, j].max()))
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)
        ax.set_title(col)
        ax.set_xlabel("true")
        ax.set_ylabel("pred")
        ax.grid(alpha=0.2)
    for k in range(len(output_columns), rows * cols):
        axes[k // cols][k % cols].axis("off")
    fig.suptitle(f"{relation} / {backend}: true vs pred")
    fig.tight_layout()
    fig.savefig(backend_dir / "scatter_true_vs_pred.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.4 * rows), squeeze=False)
    for j, col in enumerate(output_columns):
        ax = axes[j // cols][j % cols]
        ax.hist(err[:, j], bins=80, alpha=0.9)
        ax.set_title(col)
        ax.set_xlabel("pred - true")
        ax.grid(alpha=0.2)
    for k in range(len(output_columns), rows * cols):
        axes[k // cols][k % cols].axis("off")
    fig.suptitle(f"{relation} / {backend}: error histogram")
    fig.tight_layout()
    fig.savefig(backend_dir / "error_hist.png", dpi=160)
    plt.close(fig)

    t = np.arange(min(time_series_points, n))
    fig, axes = plt.subplots(rows, cols, figsize=(4.8 * cols, 3.2 * rows), squeeze=False)
    for j, col in enumerate(output_columns):
        ax = axes[j // cols][j % cols]
        ax.plot(t, y[: len(t), j], label="true", linewidth=1.2)
        ax.plot(t, pred[: len(t), j], label="pred", linewidth=1.0, alpha=0.85)
        ax.set_title(col)
        ax.grid(alpha=0.2)
        if j == 0:
            ax.legend()
    for k in range(len(output_columns), rows * cols):
        axes[k // cols][k % cols].axis("off")
    fig.suptitle(f"{relation} / {backend}: first {len(t)} samples")
    fig.tight_layout()
    fig.savefig(backend_dir / "timeseries.png", dpi=160)
    plt.close(fig)


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = ["relation", "backend", "source", "output", "count", "mae", "rmse", "max_abs", "bias"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


def write_html_report(output_dir: Path, rows: list[dict], relations: list[str]) -> None:
    def fmt(value) -> str:
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    parts = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        "<title>PR/UL model evaluation</title>",
        "<style>",
        "body{font-family:system-ui,sans-serif;margin:24px;color:#222;background:#f7f8fa}",
        "h1{font-size:24px} h2{margin-top:34px}",
        "table{border-collapse:collapse;background:white;margin:12px 0;width:100%}",
        "th,td{border:1px solid #d0d7de;padding:6px 8px;text-align:right}",
        "th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}",
        "img{max-width:100%;border:1px solid #d0d7de;background:white;margin:8px 0 20px}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}",
        ".card{background:white;border:1px solid #d0d7de;padding:12px}",
        "</style></head><body>",
        "<h1>PR/UL Model Evaluation</h1>",
    ]
    for relation in relations:
        relation_rows = [r for r in rows if r["relation"] == relation and r["source"] == "all" and r["output"] == "__all__"]
        if not relation_rows:
            continue
        parts.append(f"<h2>{relation}</h2>")
        parts.append("<table><thead><tr><th>backend</th><th>count</th><th>MAE</th><th>RMSE</th><th>MaxAbs</th><th>Bias</th></tr></thead><tbody>")
        for row in relation_rows:
            parts.append(
                "<tr>"
                f"<td>{row['backend']}</td>"
                f"<td>{row['count']}</td>"
                f"<td>{fmt(row['mae'])}</td>"
                f"<td>{fmt(row['rmse'])}</td>"
                f"<td>{fmt(row['max_abs'])}</td>"
                f"<td>{fmt(row['bias'])}</td>"
                "</tr>"
            )
        parts.append("</tbody></table>")
        overall = Path(relation) / "overall_mae_rmse.png"
        if (output_dir / overall).exists():
            parts.append(f"<img src='{overall.as_posix()}' alt='{relation} overall'>")
        parts.append("<div class='grid'>")
        for row in relation_rows:
            backend = row["backend"]
            backend_dir = Path(relation) / backend
            if not (output_dir / backend_dir).exists():
                continue
            parts.append(f"<div class='card'><h3>{backend}</h3>")
            for filename in ("scatter_true_vs_pred.png", "error_hist.png", "timeseries.png"):
                path = backend_dir / filename
                if (output_dir / path).exists():
                    parts.append(f"<a href='{path.as_posix()}'><img src='{path.as_posix()}' alt='{backend} {filename}'></a>")
            parts.append("</div>")
        parts.append("</div>")
    parts.append("</body></html>")
    (output_dir / "report.html").write_text("\n".join(parts), encoding="utf-8")


def evaluate(args: argparse.Namespace) -> None:
    maybe_reexec_for_plotting()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt = setup_matplotlib(output_dir)

    csv_paths = resolve_csv_paths(args.csv)
    relations = [item.strip() for item in args.relations.split(",") if item.strip()]
    backends = [item.strip() for item in args.backends.split(",") if item.strip()]
    summary_rows: list[dict] = []
    failures: list[dict] = []

    for relation in relations:
        if relation not in RELATION_COLUMNS:
            raise ValueError(f"Unknown relation: {relation}")
        input_columns, output_columns = RELATION_COLUMNS[relation]
        print(f"\n[RELATION] {relation}")
        x, y, source, _file_ids = load_relation_data(
            csv_paths,
            input_columns,
            output_columns,
            args.max_abs_tau,
            args.max_abs_vel,
            args.max_rows_per_file,
        )
        relation_dir = output_dir / relation
        relation_dir.mkdir(parents=True, exist_ok=True)
        relation_summary: list[dict] = []

        for backend in backends:
            if backend == "old_pkl" and relation not in OLD_PKL_RELATIONS:
                print(f"[SKIP] {backend} / {relation}: old_pkl has state models only")
                continue
            try:
                if backend == "weighted_mlp":
                    pred = predict_weighted(Path(args.weighted_mlp_dir), relation, x, args.batch_size)
                elif backend == "poly_degree3":
                    pred = predict_poly(Path(args.poly_dir), relation, x)
                elif backend == "old_pkl":
                    pred = predict_old_pkl_external(
                        Path(args.old_pkl_dir),
                        relation,
                        x,
                        output_dir / "_tmp",
                        Path(args.sklearn_python),
                    )
                else:
                    raise ValueError(f"Unknown backend: {backend}")
                metrics = compute_metrics(y, pred, output_columns, source)
                for row in metrics:
                    row["relation"] = relation
                    row["backend"] = backend
                summary_rows.extend(metrics)
                relation_summary.extend(metrics)
                overall = next(r for r in metrics if r["source"] == "all" and r["output"] == "__all__")
                print(
                    f"[OK] {backend}: count={overall['count']} "
                    f"MAE={overall['mae']:.6g} RMSE={overall['rmse']:.6g} MaxAbs={overall['max_abs']:.6g}"
                )
                if args.plots:
                    plot_prediction_details(
                        plt,
                        relation_dir,
                        relation,
                        backend,
                        output_columns,
                        y,
                        pred,
                        args.max_plot_points,
                        args.time_series_points,
                    )
            except Exception as exc:
                failures.append({"relation": relation, "backend": backend, "error": str(exc)})
                print(f"[FAIL] {backend} / {relation}: {exc}")

        if args.plots:
            plot_backend_summary(plt, relation_dir, relation, relation_summary)

    write_summary_csv(output_dir / "summary.csv", summary_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    (output_dir / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    write_html_report(output_dir, summary_rows, relations)
    print(f"\n[WROTE] {output_dir / 'summary.csv'}")
    print(f"[WROTE] {output_dir / 'summary.json'}")
    print(f"[WROTE] {output_dir / 'report.html'}")
    print(f"[WROTE] plots under {output_dir}")
    if failures:
        print(f"[WARN] {len(failures)} backend/relation combinations failed; see failures.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate and visualize PR/UL ankle relation models.")
    parser.add_argument("--csv", default=str(DEFAULT_DATA), help="CSV file, directory, or comma-separated CSV list.")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--weighted_mlp_dir", default=str(DEFAULT_WEIGHTED_MLP_DIR))
    parser.add_argument("--poly_dir", default=str(DEFAULT_POLY_DIR))
    parser.add_argument("--old_pkl_dir", default=str(DEFAULT_OLD_PKL_DIR))
    parser.add_argument("--sklearn_python", default=str(DEFAULT_SKLEARN_PYTHON))
    parser.add_argument("--relations", default="pr_to_ul_state,ul_to_pr_state,pr_to_ul_torque,ul_to_pr_torque,pr_to_ul_full,ul_to_pr_full")
    parser.add_argument("--backends", default="weighted_mlp,poly_degree3,old_pkl")
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--max_abs_tau", type=float, default=120.0)
    parser.add_argument("--max_abs_vel", type=float, default=80.0)
    parser.add_argument("--max_rows_per_file", type=int, default=None)
    parser.add_argument("--plots", action="store_true", default=True)
    parser.add_argument("--no_plots", action="store_false", dest="plots")
    parser.add_argument("--max_plot_points", type=int, default=6000)
    parser.add_argument("--time_series_points", type=int, default=700)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
