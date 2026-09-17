"""Train weighted MLP relations between pitch/roll and upper/lower ankles.

This trainer is meant for mixed data sources:

* policy data: free-base policy motion.  Torque columns come from simulated PR
  actuator force and virtual-work mapping to UL, so torque labels are preferred.
* suspended data: base-held ankle sweeps.  Position/velocity coverage is useful,
  while UL torque columns can be selected from equivalent or model-formula values
  and are down-weighted by default.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


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

UL_TORQUE_SUFFIX = {
    "canonical": "",
    "equiv": "_equiv",
    "equiv_from_pd": "_equiv_from_pd",
    "calc_model": "_calc_model",
    "calc_model_unclipped": "_calc_model_unclipped",
}


@dataclass
class MLPConfig:
    input_dim: int
    output_dim: int
    hidden_dim: int = 192
    num_hidden_layers: int = 4
    activation: str = "silu"
    dropout: float = 0.0


class RelationMLP(nn.Module):
    def __init__(self, config: MLPConfig):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = config.input_dim
        for _ in range(config.num_hidden_layers):
            layers.append(nn.Linear(in_dim, config.hidden_dim))
            layers.append(make_activation(config.activation))
            if config.dropout > 0.0:
                layers.append(nn.Dropout(config.dropout))
            in_dim = config.hidden_dim
        layers.append(nn.Linear(in_dim, config.output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ScaledRelationMLP(nn.Module):
    def __init__(self, model: RelationMLP, input_mean, input_std, output_mean, output_std):
        super().__init__()
        self.model = model
        self.register_buffer("input_mean", input_mean)
        self.register_buffer("input_std", input_std)
        self.register_buffer("output_mean", output_mean)
        self.register_buffer("output_std", output_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = (x - self.input_mean) / self.input_std
        y_norm = self.model(x_norm)
        return y_norm * self.output_std + self.output_mean


def make_activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "softsign":
        return nn.Softsign()
    if name == "silu":
        return nn.SiLU()
    if name == "elu":
        return nn.ELU()
    raise ValueError(f"Unsupported activation: {name}")


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


def infer_source(path: Path, df: pd.DataFrame) -> np.ndarray:
    if "source" in df.columns:
        raw = df["source"].astype(str).str.lower()
    else:
        raw = pd.Series([path.stem.lower()] * len(df), index=df.index)
    out = np.full(len(df), "other", dtype=object)
    out[raw.str.contains("policy", na=False).to_numpy()] = "policy"
    out[raw.str.contains("suspended|base-held|base_held", na=False).to_numpy()] = "suspended"
    return out


def maybe_select_suspended_ul_torque(df: pd.DataFrame, source: np.ndarray, mode: str) -> pd.DataFrame:
    suffix = UL_TORQUE_SUFFIX[mode]
    if not suffix:
        return df
    df = df.copy()
    suspended_mask = source == "suspended"
    for col in UL_TAU:
        alt = f"{col}{suffix}"
        if alt in df.columns:
            df.loc[suspended_mask, col] = df.loc[suspended_mask, alt]
    return df


def output_group_weight(column: str, args: argparse.Namespace) -> float:
    if column.endswith("_tau"):
        return args.tau_output_weight
    if column.endswith("_vel"):
        return args.vel_output_weight
    return args.pos_output_weight


def make_weight_matrix(source: np.ndarray, output_columns: list[str], args: argparse.Namespace) -> np.ndarray:
    weights = np.ones((len(source), len(output_columns)), dtype=np.float32)
    for j, col in enumerate(output_columns):
        weights[:, j] *= output_group_weight(col, args)
        if col.endswith("_tau"):
            weights[source == "policy", j] *= args.policy_torque_weight
            weights[source == "suspended", j] *= args.suspended_torque_weight
            weights[source == "other", j] *= args.other_torque_weight
        else:
            weights[source == "policy", j] *= args.policy_state_weight
            weights[source == "suspended", j] *= args.suspended_state_weight
            weights[source == "other", j] *= args.other_state_weight
    return weights


def filter_rows(df: pd.DataFrame, columns: list[str], args: argparse.Namespace) -> pd.DataFrame:
    data = df[columns].replace([np.inf, -np.inf], np.nan).dropna()
    if args.max_abs_tau is not None:
        tau_cols = [col for col in data.columns if col.endswith("_tau")]
        if tau_cols:
            data = data[(data[tau_cols].abs() <= args.max_abs_tau).all(axis=1)]
    if args.max_abs_vel is not None:
        vel_cols = [col for col in data.columns if col.endswith("_vel")]
        if vel_cols:
            data = data[(data[vel_cols].abs() <= args.max_abs_vel).all(axis=1)]
    return data


def split_indices(n: int, train_ratio: float, val_ratio: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.")
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    idx = np.arange(n)
    return idx[:train_end], idx[train_end:val_end], idx[val_end:]


def load_chunks(args: argparse.Namespace, input_columns: list[str], output_columns: list[str]):
    paths = resolve_csv_paths(args.csv)
    chunks = []
    for path in paths:
        df_raw = pd.read_csv(path)
        source_full = infer_source(path, df_raw)
        df = maybe_select_suspended_ul_torque(df_raw, source_full, args.suspended_torque_mode)
        missing = [col for col in input_columns + output_columns if col not in df.columns]
        if missing:
            print(f"skip {path}: missing columns {missing}")
            continue

        all_cols = list(dict.fromkeys(input_columns + output_columns))
        kept = filter_rows(df, all_cols, args)
        if len(kept) == 0:
            print(f"loaded 0 usable rows: {path}")
            continue
        source = source_full[kept.index.to_numpy()]
        x = kept[input_columns].to_numpy(dtype=np.float32)
        y = kept[output_columns].to_numpy(dtype=np.float32)
        w = make_weight_matrix(source, output_columns, args)
        chunks.append({"path": path, "x": x, "y": y, "w": w, "source": source})
        counts = {name: int((source == name).sum()) for name in ("policy", "suspended", "other")}
        print(f"loaded {len(kept)} rows: {path} source_counts={counts}")
    if not chunks:
        raise ValueError("No usable CSV data after filtering.")
    return chunks, paths


def concat_split(chunks, train_ratio: float, val_ratio: float):
    parts = {"train": [], "val": [], "test": []}
    for chunk in chunks:
        train_i, val_i, test_i = split_indices(len(chunk["x"]), train_ratio, val_ratio)
        for name, idx in (("train", train_i), ("val", val_i), ("test", test_i)):
            parts[name].append(
                {
                    "x": chunk["x"][idx],
                    "y": chunk["y"][idx],
                    "w": chunk["w"][idx],
                    "source": chunk["source"][idx],
                }
            )

    out = {}
    for name, items in parts.items():
        out[name] = {
            "x": np.vstack([item["x"] for item in items]).astype(np.float32),
            "y": np.vstack([item["y"] for item in items]).astype(np.float32),
            "w": np.vstack([item["w"] for item in items]).astype(np.float32),
            "source": np.concatenate([item["source"] for item in items]),
        }
    return out


def make_loader(x: np.ndarray, y: np.ndarray, w: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(w))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def weighted_mse(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    loss = (pred - target).square() * weight
    return loss.sum() / torch.clamp(weight.sum(), min=1.0)


def evaluate(model: ScaledRelationMLP, split: dict, output_columns: list[str], batch_size: int, device: torch.device):
    x_raw = split["x"]
    y_raw = split["y"]
    source = split["source"]
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(x_raw), batch_size):
            xb = torch.from_numpy(x_raw[start : start + batch_size]).to(device)
            preds.append(model(xb).cpu().numpy())
    pred = np.vstack(preds)
    err = pred - y_raw

    def metrics(mask: np.ndarray):
        if not np.any(mask):
            return None
        e = err[mask]
        return {
            "count": int(mask.sum()),
            "mae_all": float(np.mean(np.abs(e))),
            "rmse_all": float(np.sqrt(np.mean(e * e))),
            "max_abs_all": float(np.max(np.abs(e))),
            "mae": np.mean(np.abs(e), axis=0),
            "rmse": np.sqrt(np.mean(e * e, axis=0)),
            "max_abs": np.max(np.abs(e), axis=0),
        }

    out = {"all": metrics(np.ones(len(source), dtype=bool))}
    for name in ("policy", "suspended", "other"):
        out[name] = metrics(source == name)
    return out


def print_metrics(title: str, metrics: dict, output_columns: list[str]) -> None:
    if metrics is None:
        return
    print(f"{title}: count={metrics['count']} MAE={metrics['mae_all']:.8f} RMSE={metrics['rmse_all']:.8f} MaxAbs={metrics['max_abs_all']:.8f}")
    for col, mae, rmse, max_abs in zip(output_columns, metrics["mae"], metrics["rmse"], metrics["max_abs"]):
        print(f"  {col}: MAE={mae:.8f}, RMSE={rmse:.8f}, MaxAbs={max_abs:.8f}")


def train(args: argparse.Namespace) -> None:
    default_inputs, default_outputs = RELATION_COLUMNS[args.relation]
    input_columns = parse_columns(args.input_columns, default_inputs)
    output_columns = parse_columns(args.output_columns, default_outputs)

    chunks, csv_paths = load_chunks(args, input_columns, output_columns)
    splits = concat_split(chunks, args.train_ratio, args.val_ratio)

    train_x_raw = splits["train"]["x"]
    train_y_raw = splits["train"]["y"]
    input_mean = train_x_raw.mean(axis=0, keepdims=True)
    input_std = train_x_raw.std(axis=0, keepdims=True) + 1e-8
    output_mean = train_y_raw.mean(axis=0, keepdims=True)
    output_std = train_y_raw.std(axis=0, keepdims=True) + 1e-8

    train_x = ((splits["train"]["x"] - input_mean) / input_std).astype(np.float32)
    train_y = ((splits["train"]["y"] - output_mean) / output_std).astype(np.float32)
    val_x = ((splits["val"]["x"] - input_mean) / input_std).astype(np.float32)
    val_y = ((splits["val"]["y"] - output_mean) / output_std).astype(np.float32)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    config = MLPConfig(
        input_dim=train_x.shape[1],
        output_dim=train_y.shape[1],
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.layers,
        activation=args.activation,
        dropout=args.dropout,
    )
    model = RelationMLP(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = make_loader(train_x, train_y, splits["train"]["w"], args.batch_size, True)
    val_loader = make_loader(val_x, val_y, splits["val"]["w"], args.batch_size, False)

    save_dir = Path(args.save_dir) / args.relation
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_dir / "checkpoint.pt"
    scripted_path = save_dir / "scripted.pth"
    metadata_path = save_dir / "metadata.json"
    if not args.overwrite:
        existing = [str(path) for path in (checkpoint_path, scripted_path, metadata_path) if path.exists()]
        if existing:
            raise FileExistsError("Refusing to overwrite existing model files: " + ", ".join(existing))

    print(f"relation={args.relation}")
    print(f"input columns: {input_columns}")
    print(f"output columns: {output_columns}")
    print(f"suspended_torque_mode={args.suspended_torque_mode}")
    print(f"samples: train={len(train_x)}, val={len(val_x)}, test={len(splits['test']['x'])}")
    print(
        "weights: "
        f"policy_torque={args.policy_torque_weight}, suspended_torque={args.suspended_torque_weight}, "
        f"policy_state={args.policy_state_weight}, suspended_state={args.suspended_state_weight}"
    )
    print(f"device={device}")

    best_val = float("inf")
    best_epoch = -1
    patience_count = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        for xb, yb, wb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            wb = wb.to(device)
            loss = weighted_mse(model(xb), yb, wb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach()) * len(xb)
            train_count += len(xb)
        train_loss /= max(train_count, 1)

        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for xb, yb, wb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                wb = wb.to(device)
                loss = weighted_mse(model(xb), yb, wb)
                val_loss += float(loss.detach()) * len(xb)
                val_count += len(xb)
        val_loss /= max(val_count, 1)

        if epoch == 1 or epoch % args.print_every == 0:
            print(f"epoch={epoch:04d} train_weighted_mse={train_loss:.8f} val_weighted_mse={val_loss:.8f}")

        if val_loss < best_val - args.min_delta:
            best_val = val_loss
            best_epoch = epoch
            patience_count = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(config),
                    "relation": args.relation,
                    "input_columns": input_columns,
                    "output_columns": output_columns,
                    "input_mean": input_mean.astype(np.float32),
                    "input_std": input_std.astype(np.float32),
                    "output_mean": output_mean.astype(np.float32),
                    "output_std": output_std.astype(np.float32),
                    "csv": [str(path) for path in csv_paths],
                    "epoch": epoch,
                    "val_weighted_mse": val_loss,
                    "trainer_args": vars(args),
                },
                checkpoint_path,
            )
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"early stopping at epoch={epoch}, best_epoch={best_epoch}, best_val={best_val:.8f}")
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    best_config = MLPConfig(**checkpoint["config"])
    best_model = RelationMLP(best_config).to(device)
    best_model.load_state_dict(checkpoint["model_state_dict"])
    scaled_model = ScaledRelationMLP(
        best_model,
        torch.tensor(checkpoint["input_mean"], dtype=torch.float32, device=device),
        torch.tensor(checkpoint["input_std"], dtype=torch.float32, device=device),
        torch.tensor(checkpoint["output_mean"], dtype=torch.float32, device=device),
        torch.tensor(checkpoint["output_std"], dtype=torch.float32, device=device),
    ).to(device)
    scaled_model.eval()
    torch.jit.script(scaled_model.cpu()).save(scripted_path)

    metrics = evaluate(scaled_model.to(device), splits["test"], output_columns, args.batch_size, device)
    print("test metrics on raw units:")
    print_metrics("all", metrics["all"], output_columns)
    print_metrics("policy", metrics["policy"], output_columns)
    print_metrics("suspended", metrics["suspended"], output_columns)
    print_metrics("other", metrics["other"], output_columns)

    metadata = {
        "relation": args.relation,
        "input_columns": input_columns,
        "output_columns": output_columns,
        "csv": [str(path) for path in csv_paths],
        "best_epoch": int(checkpoint["epoch"]),
        "best_val_weighted_mse": float(checkpoint["val_weighted_mse"]),
        "suspended_torque_mode": args.suspended_torque_mode,
        "weights": {
            "policy_torque_weight": args.policy_torque_weight,
            "suspended_torque_weight": args.suspended_torque_weight,
            "policy_state_weight": args.policy_state_weight,
            "suspended_state_weight": args.suspended_state_weight,
            "pos_output_weight": args.pos_output_weight,
            "vel_output_weight": args.vel_output_weight,
            "tau_output_weight": args.tau_output_weight,
        },
        "test_metrics": {
            key: None
            if value is None
            else {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in value.items()}
            for key, value in metrics.items()
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"saved checkpoint: {checkpoint_path}")
    print(f"saved scripted model: {scripted_path}")
    print(f"saved metadata: {metadata_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train weighted PR/UL ankle relation MLPs.")
    parser.add_argument("--csv", required=True, help="CSV file, directory, or comma-separated CSV list.")
    parser.add_argument("--relation", choices=tuple(RELATION_COLUMNS), default="pr_to_ul_full")
    parser.add_argument("--save-dir", default=str(Path(__file__).resolve().parent / "models_pr_ul_relation_weighted"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--input-columns", default=None)
    parser.add_argument("--output-columns", default=None)
    parser.add_argument("--suspended-torque-mode", choices=tuple(UL_TORQUE_SUFFIX), default="equiv")
    parser.add_argument("--policy-torque-weight", type=float, default=6.0)
    parser.add_argument("--suspended-torque-weight", type=float, default=0.75)
    parser.add_argument("--other-torque-weight", type=float, default=1.0)
    parser.add_argument("--policy-state-weight", type=float, default=1.0)
    parser.add_argument("--suspended-state-weight", type=float, default=1.0)
    parser.add_argument("--other-state-weight", type=float, default=1.0)
    parser.add_argument("--pos-output-weight", type=float, default=1.0)
    parser.add_argument("--vel-output-weight", type=float, default=0.7)
    parser.add_argument("--tau-output-weight", type=float, default=1.0)
    parser.add_argument("--max-abs-tau", type=float, default=120.0)
    parser.add_argument("--max-abs-vel", type=float, default=80.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--activation", default="silu", choices=("relu", "tanh", "softsign", "silu", "elu"))
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--min-delta", type=float, default=1e-7)
    parser.add_argument("--print-every", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
