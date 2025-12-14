"""Baseline sequence model training script for the K-League challenge."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from sequence_dataset import EventSequenceDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a baseline sequence model")
    parser.add_argument("--features", type=Path, default=Path("artifacts/features/train_features.parquet"), help="Input engineered dataset")
    parser.add_argument("--format", choices=["parquet", "csv", "auto"], default="auto", help="Input format override")
    parser.add_argument("--sequence-length", type=int, default=10, help="Events per sequence window")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--output", type=Path, default=Path("artifacts/models/baseline"))
    parser.add_argument("--sample-rows", type=int, default=None, help="Optional sample size for quick runs")
    return parser.parse_args()


def load_features(path: Path, fmt: str, sample_rows: int | None) -> pd.DataFrame:
    fmt = fmt if fmt != "auto" else path.suffix.lstrip(".")
    if fmt == "parquet":
        df = pd.read_parquet(path)
    elif fmt == "csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported format: {fmt}")
    if sample_rows and len(df) > sample_rows:
        df = df.sample(n=sample_rows, random_state=42).sort_index()
    return df


def train_val_split(dataset: EventSequenceDataset, val_ratio: float) -> Tuple[EventSequenceDataset, EventSequenceDataset]:
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    return random_split(dataset, [train_size, val_size])


def euclidean_distance(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.sum((pred - target) ** 2, dim=-1))


class BiLSTMRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_distance = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        inputs = batch.inputs.to(DEVICE)
        targets = batch.targets.to(DEVICE)
        optimizer.zero_grad()
        preds = model(inputs)
        loss = criterion(preds, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        total_distance += euclidean_distance(preds, targets).mean().item() * inputs.size(0)
    return total_loss / len(loader.dataset), total_distance / len(loader.dataset)


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_distance = 0.0
    with torch.no_grad():
        for batch in tqdm(loader, desc="val", leave=False):
            inputs = batch.inputs.to(DEVICE)
            targets = batch.targets.to(DEVICE)
            preds = model(inputs)
            loss = criterion(preds, targets)
            total_loss += loss.item() * inputs.size(0)
            total_distance += euclidean_distance(preds, targets).mean().item() * inputs.size(0)
    return total_loss / len(loader.dataset), total_distance / len(loader.dataset)


def main() -> None:
    args = parse_args()
    df = load_features(args.features, args.format, args.sample_rows)
    dataset = EventSequenceDataset(df, sequence_length=args.sequence_length)
    if len(dataset) == 0:
        raise ValueError("No training sequences could be generated.")

    train_ds, val_ds = train_val_split(dataset, args.val_ratio)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = BiLSTMRegressor(input_dim=dataset.inputs.shape[-1]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()

    args.output.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_dist = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_dist = evaluate(model, val_loader, criterion)
        print(f"Epoch {epoch:02d}: train loss {train_loss:.4f}, val loss {val_loss:.4f}, val dist {val_dist:.4f}")
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_distance": train_dist,
            "val_loss": val_loss,
            "val_distance": val_dist,
        })

    torch.save(model.state_dict(), args.output / "model.pt")
    with open(args.output / "history.json", "w", encoding="utf-8") as fp:
        json.dump(history, fp, indent=2)

    print(f"[INFO] Training complete. Artifacts saved to {args.output}")


if __name__ == "__main__":
    main()

