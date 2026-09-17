"""Train MLP relations between pitch/roll ankles and upper/lower ankle motors.

The expected CSV comes from collect_upper_lower_ground_stand_data.py or
collect_mujoco_torque_data.py.  For ground-contact torque relations, prefer the
ground-stand collector because it also writes equivalent pitch/roll torques:

    left_pitch_tau, left_roll_tau, right_pitch_tau, right_roll_tau

Model choices:
    pr_to_ul_state   PR pos/vel -> UL pos/vel
    ul_to_pr_state   UL pos/vel -> PR pos/vel
    pr_to_ul_torque  PR pos/vel/tau -> UL tau
    ul_to_pr_torque  UL pos/vel/tau -> PR tau
    pr_to_ul_full    PR pos/vel/tau -> UL pos/vel/tau
    ul_to_pr_full    UL pos/vel/tau -> PR pos/vel/tau
"""

from __future__ import annotations

import argparse
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


@dataclass
class MLPConfig:
    input_dim: int
    output_dim: int
    hidden_dim: int = 128
    num_hidden_layers: int = 3
    activation: str = "softsign"
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
    def __init__(
        self,
        model: RelationMLP,
        input_mean: torch.Tensor,
        input_std: torch.Tensor,
        output_mean: torch.Tensor,
        output_std: torch.Tensor,
    ):
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


def load_dataset(
    csv_arg: str,
    input_columns: list[str],
    output_columns: list[str],
    experiment_filter: set[str] | None,
    max_abs_tau: float | None,
) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    xs = []
    ys = []
    csv_paths = resolve_csv_paths(csv_arg)
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        if experiment_filter is not None and "experiment" in df.columns:
            df = df[df["experiment"].astype(str).isin(experiment_filter)]
        missing = [col for col in input_columns + output_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in {csv_path}: {missing}")
        data = df[input_columns + output_columns].replace([np.inf, -np.inf], np.nan).dropna()
        if max_abs_tau is not None:
            tau_cols = [col for col in data.columns if col.endswith("_tau")]
            if tau_cols:
                data = data[(data[tau_cols].abs() <= max_abs_tau).all(axis=1)]
        if len(data) == 0:
            print(f"loaded 0 usable rows: {csv_path}")
            continue
        xs.append(data[input_columns].to_numpy(dtype=np.float32))
        ys.append(data[output_columns].to_numpy(dtype=np.float32))
        print(f"loaded {len(data)} rows: {csv_path}")
    if not xs:
        raise ValueError("No usable rows after filtering.")
    return np.vstack(xs), np.vstack(ys), csv_paths


def split_temporal(x: np.ndarray, y: np.ndarray, train_ratio: float, val_ratio: float):
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.")
    n = len(x)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return (x[:train_end], y[:train_end]), (x[train_end:val_end], y[train_end:val_end]), (x[val_end:], y[val_end:])


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=batch_size, shuffle=shuffle)


def evaluate(model: ScaledRelationMLP, x_raw: np.ndarray, y_raw: np.ndarray, batch_size: int, device: torch.device):
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(x_raw), batch_size):
            xb = torch.from_numpy(x_raw[start : start + batch_size]).to(device)
            preds.append(model(xb).cpu().numpy())
    pred = np.vstack(preds)
    err = pred - y_raw
    return {
        "mae_all": float(np.mean(np.abs(err))),
        "rmse_all": float(np.sqrt(np.mean(err * err))),
        "max_abs_all": float(np.max(np.abs(err))),
        "mae": np.mean(np.abs(err), axis=0),
        "rmse": np.sqrt(np.mean(err * err, axis=0)),
        "max_abs": np.max(np.abs(err), axis=0),
    }


def train(args: argparse.Namespace) -> None:
    default_inputs, default_outputs = RELATION_COLUMNS[args.relation]
    input_columns = parse_columns(args.input_columns, default_inputs)
    output_columns = parse_columns(args.output_columns, default_outputs)
    experiment_filter = None
    if args.experiments:
        experiment_filter = {item.strip() for item in args.experiments.split(",") if item.strip()}

    save_dir = Path(args.save_dir) / args.relation
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_dir / "checkpoint.pt"
    scripted_path = save_dir / "scripted.pth"
    if not args.overwrite:
        existing = [str(path) for path in (checkpoint_path, scripted_path) if path.exists()]
        if existing:
            raise FileExistsError("Refusing to overwrite existing model files: " + ", ".join(existing))

    x_raw, y_raw, csv_paths = load_dataset(args.csv, input_columns, output_columns, experiment_filter, args.max_abs_tau)
    (train_x_raw, train_y_raw), (val_x_raw, val_y_raw), (test_x_raw, test_y_raw) = split_temporal(
        x_raw,
        y_raw,
        args.train_ratio,
        args.val_ratio,
    )

    input_mean = train_x_raw.mean(axis=0, keepdims=True)
    input_std = train_x_raw.std(axis=0, keepdims=True) + 1e-8
    output_mean = train_y_raw.mean(axis=0, keepdims=True)
    output_std = train_y_raw.std(axis=0, keepdims=True) + 1e-8
    train_x = ((train_x_raw - input_mean) / input_std).astype(np.float32)
    train_y = ((train_y_raw - output_mean) / output_std).astype(np.float32)
    val_x = ((val_x_raw - input_mean) / input_std).astype(np.float32)
    val_y = ((val_y_raw - output_mean) / output_std).astype(np.float32)

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
    loss_fn = nn.MSELoss()
    train_loader = make_loader(train_x, train_y, args.batch_size, True)
    val_loader = make_loader(val_x, val_y, args.batch_size, False)

    best_val = float("inf")
    best_epoch = -1
    patience_count = 0
    print(f"relation={args.relation}")
    print(f"csv files={len(csv_paths)}")
    print(f"samples: train={len(train_x_raw)}, val={len(val_x_raw)}, test={len(test_x_raw)}")
    print(f"input columns: {input_columns}")
    print(f"output columns: {output_columns}")
    print(f"device={device}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            loss = loss_fn(model(xb), yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach()) * len(xb)
            train_count += len(xb)

        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                loss = loss_fn(model(xb), yb)
                val_loss += float(loss.detach()) * len(xb)
                val_count += len(xb)
        train_loss /= max(train_count, 1)
        val_loss /= max(val_count, 1)

        if epoch == 1 or epoch % args.print_every == 0:
            print(f"epoch={epoch:04d} train_mse={train_loss:.8f} val_mse={val_loss:.8f}")

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
                    "val_mse": val_loss,
                },
                checkpoint_path,
            )
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"early stopping at epoch={epoch}, best_epoch={best_epoch}, best_val_mse={best_val:.8f}")
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

    metrics = evaluate(scaled_model.to(device), test_x_raw, test_y_raw, args.batch_size, device)
    print("test metrics on raw units:")
    print(f"  MAE all:     {metrics['mae_all']:.8f}")
    print(f"  RMSE all:    {metrics['rmse_all']:.8f}")
    print(f"  Max abs all: {metrics['max_abs_all']:.8f}")
    for col, mae, rmse, max_abs in zip(output_columns, metrics["mae"], metrics["rmse"], metrics["max_abs"]):
        print(f"  {col}: MAE={mae:.8f}, RMSE={rmse:.8f}, MaxAbs={max_abs:.8f}")
    print(f"saved checkpoint: {checkpoint_path}")
    print(f"saved scripted model: {scripted_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train pitch/roll <-> upper/lower ankle relation MLPs.")
    parser.add_argument("--csv", required=True, help="CSV file, directory of CSV files, or comma-separated CSV list.")
    parser.add_argument("--relation", choices=tuple(RELATION_COLUMNS), default="pr_to_ul_full")
    parser.add_argument("--save-dir", default=str(Path(__file__).resolve().parent / "models_pr_ul_relation"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--experiments", default=None, help="Optional comma-separated experiment names to keep.")
    parser.add_argument("--max-abs-tau", type=float, default=None, help="Drop rows where any selected *_tau column exceeds this value.")
    parser.add_argument("--input-columns", default=None, help="Comma-separated input columns. Overrides --relation defaults.")
    parser.add_argument("--output-columns", default=None, help="Comma-separated output columns. Overrides --relation defaults.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--activation", default="softsign", choices=("relu", "tanh", "softsign", "silu", "elu"))
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--min-delta", type=float, default=1e-7)
    parser.add_argument("--print-every", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
