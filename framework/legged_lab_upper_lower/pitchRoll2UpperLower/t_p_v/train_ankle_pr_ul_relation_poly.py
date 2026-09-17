"""Train polynomial ridge relations between pitch/roll and upper/lower ankles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


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


def parse_columns(value: str | None, defaults: Sequence[str]) -> list[str]:
    if not value:
        return list(defaults)
    return [item.strip() for item in value.split(",") if item.strip()]


def resolve_csv_paths(csv_arg: str) -> list[Path]:
    paths: list[Path] = []
    for item in csv_arg.split(","):
        item = item.strip()
        if not item:
            continue
        path = Path(item)
        if path.is_dir():
            paths.extend(sorted(path.glob("*.csv")))
        else:
            paths.append(path)
    if not paths:
        raise ValueError("No CSV files were provided.")
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing CSV files: " + ", ".join(missing))
    return paths


def load_dataset(
    csv_arg: str,
    input_columns: list[str],
    output_columns: list[str],
    max_abs_tau: float | None,
) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    xs = []
    ys = []
    paths = resolve_csv_paths(csv_arg)
    for path in paths:
        df = pd.read_csv(path)
        missing = [col for col in input_columns + output_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in {path}: {missing}")
        data = df[input_columns + output_columns].replace([np.inf, -np.inf], np.nan).dropna()
        if max_abs_tau is not None:
            tau_cols = [col for col in data.columns if col.endswith("_tau")]
            if tau_cols:
                data = data[(data[tau_cols].abs() <= max_abs_tau).all(axis=1)]
        if len(data) == 0:
            print(f"loaded 0 usable rows: {path}")
            continue
        xs.append(data[input_columns].to_numpy(dtype=np.float64))
        ys.append(data[output_columns].to_numpy(dtype=np.float64))
        print(f"loaded {len(data)} rows: {path}")
    if not xs:
        raise ValueError("No usable rows after filtering.")
    return np.vstack(xs), np.vstack(ys), paths


def split_temporal(x: np.ndarray, y: np.ndarray, train_ratio: float):
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("--train-ratio must be in (0, 1).")
    split = int(len(x) * train_ratio)
    return x[:split], y[:split], x[split:], y[split:]


def export_pipeline_npz(model, input_columns: list[str], output_columns: list[str], save_path: Path, metadata: dict) -> None:
    scaler = model.named_steps["standardscaler"]
    poly = model.named_steps["polynomialfeatures"]
    ridge = model.named_steps["ridge"]
    np.savez_compressed(
        save_path,
        input_columns=np.asarray(input_columns),
        output_columns=np.asarray(output_columns),
        input_mean=scaler.mean_.astype(np.float32),
        input_scale=scaler.scale_.astype(np.float32),
        powers=poly.powers_.astype(np.int64),
        coef=ridge.coef_.astype(np.float32),
        intercept=ridge.intercept_.astype(np.float32),
        metadata_json=np.asarray(json.dumps(metadata, indent=2)),
    )


def train(args: argparse.Namespace) -> None:
    default_inputs, default_outputs = RELATION_COLUMNS[args.relation]
    input_columns = parse_columns(args.input_columns, default_inputs)
    output_columns = parse_columns(args.output_columns, default_outputs)
    x, y, csv_paths = load_dataset(args.csv, input_columns, output_columns, args.max_abs_tau)
    train_x, train_y, test_x, test_y = split_temporal(x, y, args.train_ratio)

    model = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=args.degree, include_bias=False),
        Ridge(alpha=args.alpha),
    )
    model.fit(train_x, train_y)
    pred = model.predict(test_x)
    err = pred - test_y
    mae = mean_absolute_error(test_y, pred, multioutput="raw_values")
    rmse = np.sqrt(mean_squared_error(test_y, pred, multioutput="raw_values"))

    save_dir = Path(args.save_dir) / args.relation
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "poly_relation.npz"
    if save_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {save_path}. Use --overwrite.")
    metadata = {
        "relation": args.relation,
        "degree": args.degree,
        "alpha": args.alpha,
        "csv": [str(path) for path in csv_paths],
        "samples": {"train": int(len(train_x)), "test": int(len(test_x))},
        "mae_all": float(np.mean(np.abs(err))),
        "rmse_all": float(np.sqrt(np.mean(err * err))),
        "max_abs_all": float(np.max(np.abs(err))),
    }
    export_pipeline_npz(model, input_columns, output_columns, save_path, metadata)

    print(f"relation={args.relation}")
    print(f"samples: train={len(train_x)}, test={len(test_x)}")
    print(f"input columns: {input_columns}")
    print(f"output columns: {output_columns}")
    print(f"MAE all: {metadata['mae_all']:.8f}")
    print(f"RMSE all: {metadata['rmse_all']:.8f}")
    print(f"Max abs all: {metadata['max_abs_all']:.8f}")
    for col, col_mae, col_rmse in zip(output_columns, mae, rmse):
        print(f"{col}: MAE={col_mae:.8f}, RMSE={col_rmse:.8f}")
    print(f"saved polynomial model: {save_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train polynomial PR/UL ankle relation models.")
    parser.add_argument("--csv", required=True, help="CSV file, directory, or comma-separated CSV list.")
    parser.add_argument("--relation", choices=tuple(RELATION_COLUMNS), default="pr_to_ul_full")
    parser.add_argument("--save-dir", default=str(Path(__file__).resolve().parent / "models_pr_ul_relation_poly"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=1e-3)
    parser.add_argument("--max-abs-tau", type=float, default=None)
    parser.add_argument("--input-columns", default=None)
    parser.add_argument("--output-columns", default=None)
    parser.add_argument("--train-ratio", type=float, default=0.85)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
