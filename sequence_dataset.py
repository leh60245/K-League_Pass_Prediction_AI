"""Utilities to build fixed-length event sequences for modeling."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

NUMERIC_COLUMNS = [
    "start_x",
    "start_y",
    "end_x",
    "end_y",
    "delta_x",
    "delta_y",
    "event_distance",
    "vel_x",
    "vel_y",
    "speed",
    "speed_clipped",
    "delta_time_prev",
    "delta_time_next",
    "dist_to_goal",
    "dist_to_goal_norm",
    "attack_progress_momentum",
    "dynamic_pressure_index",
    "pitch_control_margin",
    "forward_control_ratio",
    "pass_availability_score",
    "off_ball_energy",
]
CATEGORICAL_COLUMNS = [
    "type_filled",
    "result_filled",
    "event_context",
    "prev_event_context",
    "start_quadrant",
    "end_quadrant",
    "type_group",
]
TARGET_COLUMNS = ["end_x", "end_y"]


def encode_categories(df: pd.DataFrame, columns: Sequence[str]) -> Tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    mappings: dict[str, dict[str, int]] = {}
    df = df.copy()
    for col in columns:
        labels = df[col].astype("category")
        mapping = {cat: idx for idx, cat in enumerate(labels.cat.categories)}
        df[col] = labels.cat.codes.replace(-1, np.nan)
        mappings[col] = mapping
    return df, mappings


def pad_sequences(sequences: List[np.ndarray], max_len: int) -> np.ndarray:
    padded = np.zeros((len(sequences), max_len, sequences[0].shape[1]), dtype=np.float32)
    for i, seq in enumerate(sequences):
        length = min(len(seq), max_len)
        padded[i, :length] = seq[-max_len:]
    return padded


@dataclass
class SequenceBatch:
    inputs: torch.Tensor
    targets: torch.Tensor
    mask: torch.Tensor


class EventSequenceDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        sequence_length: int = 10,
        numeric_cols: Sequence[str] | None = None,
        categorical_cols: Sequence[str] | None = None,
    ) -> None:
        self.sequence_length = sequence_length
        numeric_cols = numeric_cols or NUMERIC_COLUMNS
        categorical_cols = categorical_cols or []

        df = df.sort_values(["game_id", "team_id", "action_id"]).reset_index(drop=True)
        df_numeric = df[numeric_cols].fillna(0.0).astype(np.float32)

        sequences: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        mask: list[np.ndarray] = []

        grouped = df.groupby(["game_id", "team_id"], sort=False)
        for _, group in grouped:
            numeric_values = group[numeric_cols].to_numpy(dtype=np.float32)
            target_values = group[TARGET_COLUMNS].to_numpy(dtype=np.float32)
            for idx in range(sequence_length, len(group)):
                seq = numeric_values[idx - sequence_length: idx]
                tgt = target_values[idx]
                sequences.append(seq)
                targets.append(tgt)
                seq_mask = np.ones(sequence_length, dtype=np.float32)
                mask.append(seq_mask)

        self.inputs = torch.from_numpy(np.stack(sequences)) if sequences else torch.empty(0)
        self.targets = torch.from_numpy(np.stack(targets)) if targets else torch.empty(0)
        self.mask = torch.from_numpy(np.stack(mask)) if mask else torch.empty(0)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> SequenceBatch:
        return SequenceBatch(self.inputs[idx], self.targets[idx], self.mask[idx])

