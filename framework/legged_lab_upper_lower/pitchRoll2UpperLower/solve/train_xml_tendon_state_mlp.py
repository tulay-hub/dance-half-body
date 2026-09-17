"""Train TorchScript state MLPs from the Lens110 XML tendon solver.

The generated model directory is compatible with the XmlTendonUpperLower AMP
config:

    models/xml_tendon_mlp/pr_to_ul_state/scripted.pth
    models/xml_tendon_mlp/ul_to_pr_state/scripted.pth
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from xml_tendon_pr_to_ul import DEFAULT_XML, Lens110XmlTendonSolver


PR_POS = ("left_pitch_pos", "left_roll_pos", "right_pitch_pos", "right_roll_pos")
PR_VEL = ("left_pitch_vel", "left_roll_vel", "right_pitch_vel", "right_roll_vel")
UL_POS = ("left_upper_pos", "left_lower_pos", "right_upper_pos", "right_lower_pos")
UL_VEL = ("left_upper_vel", "left_lower_vel", "right_upper_vel", "right_lower_vel")


@dataclass
class MLPConfig:
    input_dim: int
    output_dim: int
    hidden_dim: int = 192
    num_hidden_layers: int = 4
    activation: str = "silu"


class RelationMLP(nn.Module):
    def __init__(self, config: MLPConfig):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = config.input_dim
        for _ in range(config.num_hidden_layers):
            layers.append(nn.Linear(in_dim, config.hidden_dim))
            layers.append(make_activation(config.activation))
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


def sample_ranges(solver: Lens110XmlTendonSolver, args: argparse.Namespace) -> tuple[float, float, float, float]:
    joint_pitch_lo, joint_pitch_hi = solver.joint_range["left_ankle_pitch_joint"]
    joint_roll_lo, joint_roll_hi = solver.joint_range["left_ankle_roll_joint"]
    pitch_lo = max(joint_pitch_lo, args.pitch_min)
    pitch_hi = min(joint_pitch_hi, args.pitch_max)
    roll_lo = max(joint_roll_lo, args.roll_min)
    roll_hi = min(joint_roll_hi, args.roll_max)
    return pitch_lo, pitch_hi, roll_lo, roll_hi


def build_side_table(
    solver: Lens110XmlTendonSolver,
    side: str,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    pitch_lo, pitch_hi, roll_lo, roll_hi = sample_ranges(solver, args)
    pitches = np.linspace(pitch_lo, pitch_hi, args.grid_pitch_samples, dtype=np.float64)
    rolls = np.linspace(roll_lo, roll_hi, args.grid_roll_samples, dtype=np.float64)
    ul = np.zeros((len(pitches), len(rolls), 2), dtype=np.float64)
    valid = np.ones((len(pitches), len(rolls)), dtype=bool)
    last = None
    start = time.time()
    count = 0
    for i, pitch in enumerate(pitches):
        for j, roll in enumerate(rolls):
            count += 1
            try:
                ul[i, j] = solver.solve_side(side, float(pitch), float(roll), initial=last)
                last = ul[i, j]
            except RuntimeError:
                valid[i, j] = False
                ul[i, j] = np.nan
                last = None
            if args.print_every_samples > 0 and count % args.print_every_samples == 0:
                rate = count / max(time.time() - start, 1e-9)
                print(f"{side}: solved {count}/{len(pitches) * len(rolls)} grid points ({rate:.1f} points/s)")
    if not valid.all():
        bad = int((~valid).sum())
        raise RuntimeError(f"{side} table has {bad} unreachable grid points; reduce pitch/roll range.")

    d_ul_d_pitch = np.gradient(ul, pitches, axis=0, edge_order=2)
    d_ul_d_roll = np.gradient(ul, rolls, axis=1, edge_order=2)
    pr = np.stack(np.meshgrid(pitches, rolls, indexing="ij"), axis=-1).reshape(-1, 2)
    jac = np.stack((d_ul_d_pitch, d_ul_d_roll), axis=-1).reshape(-1, 2, 2)
    return {"pr": pr, "ul": ul.reshape(-1, 2), "jac": jac}


def sample_pr_state(solver: Lens110XmlTendonSolver, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(args.seed)
    pitch_lo, pitch_hi, roll_lo, roll_hi = sample_ranges(solver, args)

    pitches = np.linspace(pitch_lo, pitch_hi, args.grid_pitch_samples, dtype=np.float64)
    rolls = np.linspace(roll_lo, roll_hi, args.grid_roll_samples, dtype=np.float64)
    grid = np.stack(np.meshgrid(pitches, rolls, indexing="ij"), axis=-1).reshape(-1, 2)
    grid_count = min(len(grid), args.grid_pairs)
    grid_left = grid[rng.choice(len(grid), size=grid_count, replace=False)]
    grid_right = grid[rng.choice(len(grid), size=grid_count, replace=False)]
    grid_pr = np.column_stack((grid_left[:, 0], grid_left[:, 1], grid_right[:, 0], grid_right[:, 1]))

    random_pr = np.column_stack(
        (
            rng.uniform(pitch_lo, pitch_hi, args.random_samples),
            rng.uniform(roll_lo, roll_hi, args.random_samples),
            rng.uniform(pitch_lo, pitch_hi, args.random_samples),
            rng.uniform(roll_lo, roll_hi, args.random_samples),
        )
    )
    pr_pos = np.vstack((grid_pr, random_pr)).astype(np.float64)
    pr_vel = rng.uniform(-args.max_abs_vel, args.max_abs_vel, size=pr_pos.shape).astype(np.float64)
    return pr_pos, pr_vel


def build_dataset(args: argparse.Namespace) -> dict[str, np.ndarray]:
    solver = Lens110XmlTendonSolver(args.xml, grid_samples=args.solver_grid_samples, length_tol=args.length_tol)
    rng = np.random.default_rng(args.seed)
    left = build_side_table(solver, "left", args)
    right = build_side_table(solver, "right", args)

    num_samples = args.grid_pairs + args.random_samples
    left_i = rng.integers(0, len(left["pr"]), size=num_samples)
    right_i = rng.integers(0, len(right["pr"]), size=num_samples)
    pr_pos = np.column_stack((left["pr"][left_i], right["pr"][right_i])).astype(np.float64)
    ul_pos = np.column_stack((left["ul"][left_i], right["ul"][right_i])).astype(np.float64)
    pr_vel = rng.uniform(-args.max_abs_vel, args.max_abs_vel, size=pr_pos.shape).astype(np.float64)
    ul_vel = np.zeros_like(pr_vel)
    ul_vel[:, :2] = np.einsum("nij,nj->ni", left["jac"][left_i], pr_vel[:, :2])
    ul_vel[:, 2:4] = np.einsum("nij,nj->ni", right["jac"][right_i], pr_vel[:, 2:4])
    if len(pr_pos) < args.min_kept_samples:
        raise RuntimeError(f"Only built {len(pr_pos)} samples; increase grid_pairs/random_samples.")
    print(f"built {len(pr_pos)} state samples from XML tendon side grids")

    return {
        "pr_state": np.concatenate((pr_pos, pr_vel), axis=-1).astype(np.float32),
        "ul_state": np.concatenate((ul_pos, ul_vel), axis=-1).astype(np.float32),
    }


def split_data(x: np.ndarray, y: np.ndarray, train_ratio: float, val_ratio: float):
    n = len(x)
    indices = np.arange(n)
    rng = np.random.default_rng(12345)
    rng.shuffle(indices)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train_i = indices[:train_end]
    val_i = indices[train_end:val_end]
    test_i = indices[val_end:]
    return (x[train_i], y[train_i]), (x[val_i], y[val_i]), (x[test_i], y[test_i])


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
        "mae": np.mean(np.abs(err), axis=0).astype(float).tolist(),
        "rmse": np.sqrt(np.mean(err * err, axis=0)).astype(float).tolist(),
        "max_abs": np.max(np.abs(err), axis=0).astype(float).tolist(),
    }


def train_relation(
    relation: str,
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    output_columns: tuple[str, ...],
    args: argparse.Namespace,
) -> dict:
    save_dir = Path(args.save_dir) / relation
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_dir / "checkpoint.pt"
    scripted_path = save_dir / "scripted.pth"

    (train_x_raw, train_y_raw), (val_x_raw, val_y_raw), (test_x_raw, test_y_raw) = split_data(
        x_raw, y_raw, args.train_ratio, args.val_ratio
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
    config = MLPConfig(8, 8, args.hidden_dim, args.layers, args.activation)
    model = RelationMLP(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()
    train_loader = make_loader(train_x, train_y, args.batch_size, True)
    val_loader = make_loader(val_x, val_y, args.batch_size, False)

    best_val = float("inf")
    best_epoch = 0
    patience_count = 0
    print(f"training {relation}: train={len(train_x_raw)} val={len(val_x_raw)} test={len(test_x_raw)} device={device}")
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
            print(f"{relation} epoch={epoch:04d} train_mse={train_loss:.8f} val_mse={val_loss:.8f}")
        if val_loss < best_val - args.min_delta:
            best_val = val_loss
            best_epoch = epoch
            patience_count = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(config),
                    "relation": relation,
                    "input_columns": list(PR_POS + PR_VEL if relation == "pr_to_ul_state" else UL_POS + UL_VEL),
                    "output_columns": list(output_columns),
                    "input_mean": input_mean.astype(np.float32),
                    "input_std": input_std.astype(np.float32),
                    "output_mean": output_mean.astype(np.float32),
                    "output_std": output_std.astype(np.float32),
                    "epoch": epoch,
                    "val_mse": val_loss,
                },
                checkpoint_path,
            )
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"{relation} early stop epoch={epoch}, best_epoch={best_epoch}, best_val_mse={best_val:.8f}")
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    best_model = RelationMLP(MLPConfig(**checkpoint["config"])).to(device)
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
    metadata = {
        "relation": relation,
        "config": checkpoint["config"],
        "input_columns": checkpoint["input_columns"],
        "output_columns": checkpoint["output_columns"],
        "best_epoch": int(checkpoint["epoch"]),
        "best_val_mse": float(checkpoint["val_mse"]),
        "test_metrics": metrics,
    }
    with (save_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"{relation} test MAE={metrics['mae_all']:.8f} RMSE={metrics['rmse_all']:.8f} MaxAbs={metrics['max_abs_all']:.8f}")
    print(f"saved {scripted_path}")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", default=str(DEFAULT_XML))
    parser.add_argument("--save-dir", default=str(Path(__file__).resolve().parent / "models" / "xml_tendon_mlp"))
    parser.add_argument("--solver-grid-samples", type=int, default=801)
    parser.add_argument("--length-tol", type=float, default=1e-8)
    parser.add_argument("--grid-pitch-samples", type=int, default=61)
    parser.add_argument("--grid-roll-samples", type=int, default=41)
    parser.add_argument("--grid-pairs", type=int, default=2501)
    parser.add_argument("--random-samples", type=int, default=16000)
    parser.add_argument("--pitch-min", type=float, default=-0.65)
    parser.add_argument("--pitch-max", type=float, default=0.35)
    parser.add_argument("--roll-min", type=float, default=-0.22)
    parser.add_argument("--roll-max", type=float, default=0.22)
    parser.add_argument("--max-abs-vel", type=float, default=8.0)
    parser.add_argument("--min-kept-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--activation", default="silu", choices=("relu", "tanh", "softsign", "silu", "elu"))
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--train-ratio", type=float, default=0.82)
    parser.add_argument("--val-ratio", type=float, default=0.09)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--min-delta", type=float, default=1e-7)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--print-every-samples", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    data = build_dataset(args)
    summary = {
        "dataset": {key: list(value.shape) for key, value in data.items()},
        "pr_to_ul_state": train_relation("pr_to_ul_state", data["pr_state"], data["ul_state"], UL_POS + UL_VEL, args),
        "ul_to_pr_state": train_relation("ul_to_pr_state", data["ul_state"], data["pr_state"], PR_POS + PR_VEL, args),
    }
    with (Path(args.save_dir) / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
