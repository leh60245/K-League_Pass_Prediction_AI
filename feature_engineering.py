"""Feature engineering pipeline for event-stream soccer data."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd

ARTIFACT_DIR = Path("artifacts") / "features"
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
GOAL_CENTER = np.array([105.0, 34.0])
TYPE_GROUP_MAP = {
    "Pass": "Pass",
    "Cross": "Pass",
    "Pass_Corner": "SetPiece",
    "Pass_Freekick": "SetPiece",
    "Goal Kick": "Keeper",
    "Carry": "Carry",
    "Shot": "Shot",
    "Shot_Freekick": "Shot",
    "Shot_Corner": "Shot",
    "Penalty Kick": "Shot",
    "Goal": "Shot",
    "Take-On": "Duel",
    "Duel": "Duel",
    "Tackle": "Defensive",
    "Interception": "Defensive",
    "Recovery": "Defensive",
    "Clearance": "Defensive",
    "Aerial Clearance": "Defensive",
    "Block": "Defensive",
    "Foul": "Foul",
    "Foul_Throw": "Foul",
    "Handball_Foul": "Foul",
    "Out": "Restart",
    "Throw-In": "Restart",
    "Pass_Corner": "Restart",
    "Shot_Corner": "Restart",
    "Parry": "Keeper",
    "Catch": "Keeper",
    "Error": "Other",
    "Own Goal": "Other",
}
DEAD_BALL_THRESHOLD = 5.0


BalanceStrategy = Literal["none", "type_weight", "undersample_pass"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Engineer features and mitigate class imbalance")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory with train/test/match_info CSVs")
    parser.add_argument("--output-dir", type=Path, default=ARTIFACT_DIR, help="Output directory for feature files")
    parser.add_argument("--sample-rows", type=int, default=None, help="Optional random row cap for faster iteration")
    parser.add_argument(
        "--balance-strategy",
        type=str,
        choices=["none", "type_weight", "undersample_pass"],
        default="none",
        help="How to address type_name imbalance",
    )
    parser.add_argument("--pass-ratio", type=float, default=0.5, help="Target Pass share when undersampling")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for sampling")
    parser.add_argument(
        "--save-format", choices=["parquet", "csv"], default="parquet", help="File format for engineered datasets"
    )
    return parser.parse_args()


def load_csv(path: Path, sample_rows: Optional[int], parse_dates: Optional[list[str]] = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Expected file not found: {path}")
    df = pd.read_csv(path, parse_dates=parse_dates)
    if sample_rows and len(df) > sample_rows:
        df = df.sample(n=sample_rows, random_state=42).sort_index()
    return df


def fill_result_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["result_filled"] = df["result_name"].fillna("NoResult")
    df["type_filled"] = df["type_name"].fillna("Unknown")
    df["type_result_combo"] = df["type_filled"] + "__" + df["result_filled"]
    df["type_group"] = df["type_filled"].map(TYPE_GROUP_MAP).fillna("Other")
    return df


def normalize_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for prefix in ("start", "end"):
        df[f"{prefix}_x_norm"] = df[f"{prefix}_x"] / PITCH_LENGTH
        df[f"{prefix}_y_norm"] = df[f"{prefix}_y"] / PITCH_LENGTH
    return df


def pitch_quadrant(x: float, y: float) -> str:
    if pd.isna(x) or pd.isna(y):
        return "unknown"
    x_bucket = int(np.clip(x // 35, 0, 2))
    y_bucket = int(np.clip(y // 34, 0, 1))
    return f"Q{x_bucket}{y_bucket}"


def engineer_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["delta_x"] = df["end_x"] - df["start_x"]
    df["delta_y"] = df["end_y"] - df["start_y"]
    df["event_distance"] = np.sqrt(df["delta_x"] ** 2 + df["delta_y"] ** 2)
    df["start_quadrant"] = [pitch_quadrant(x, y) for x, y in zip(df["start_x"], df["start_y"])]
    df["end_quadrant"] = [pitch_quadrant(x, y) for x, y in zip(df["end_x"], df["end_y"])]
    return df


def engineer_temporal(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["game_id", "game_episode", "action_id"]).copy()
    grp = df.groupby("game_episode")
    df["event_idx"] = grp.cumcount()
    df["episode_event_count"] = grp["action_id"].transform("count")
    df["episode_progress_ratio"] = df["event_idx"] / df["episode_event_count"].clip(lower=1)
    df["delta_time_prev"] = grp["time_seconds"].diff().fillna(0.0)
    df["delta_time_next"] = grp["time_seconds"].diff(-1).abs().fillna(0.0)
    df["dead_ball_flag"] = df["delta_time_prev"].ge(DEAD_BALL_THRESHOLD).astype(int)
    df["prev_type_name"] = grp["type_name"].shift(1).fillna("START")
    df["prev_result_filled"] = grp["result_name"].shift(1).fillna("START")
    return df


def engineer_velocity_and_direction(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["game_id", "game_episode", "action_id"]).copy()
    grp = df.groupby("game_episode")
    dx = grp["start_x"].diff()
    dy = grp["start_y"].diff()
    dt = grp["time_seconds"].diff().clip(lower=1e-3).fillna(1e-3)
    df["vel_x"] = dx / dt
    df["vel_y"] = dy / dt
    df["speed"] = np.sqrt(df["vel_x"] ** 2 + df["vel_y"] ** 2)
    df["travel_angle_rad"] = np.arctan2(dy, dx).fillna(0.0)
    df["travel_angle_deg"] = np.degrees(df["travel_angle_rad"])
    df["speed_clipped"] = df["speed"].clip(upper=20.0)
    return df


def engineer_relative_goal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    start_points = df[["start_x", "start_y"]].to_numpy()
    goal_vectors = GOAL_CENTER - start_points
    df["dist_to_goal"] = np.linalg.norm(goal_vectors, axis=1)
    df["angle_to_goal_rad"] = np.arctan2(goal_vectors[:, 1], goal_vectors[:, 0])
    df["angle_to_goal_deg"] = np.degrees(df["angle_to_goal_rad"])
    df["dist_to_goal_norm"] = df["dist_to_goal"] / PITCH_LENGTH
    return df


def engineer_context_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["event_context"] = df["type_filled"] + "__" + df["result_filled"]
    df["prev_event_context"] = df.groupby("game_episode")["event_context"].shift(1).fillna("START__START")
    return df


def compute_attack_momentum(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    movement = df[["delta_x", "delta_y"]].fillna(0.0).to_numpy()
    end_positions = df[["end_x", "end_y"]].ffill().fillna(0.0).to_numpy()
    goal_vectors = GOAL_CENTER - end_positions
    norms = np.linalg.norm(goal_vectors, axis=1)
    norms[norms == 0] = 1.0
    goal_unit = goal_vectors / norms[:, None]
    df["attack_progress_momentum"] = np.sum(movement * goal_unit, axis=1)
    return df


def _gaussian_weights(center: np.ndarray, points: list[tuple[float, float]], sigma: float) -> float:
    if not len(points):
        return 0.0
    pts = np.array(points, dtype=float)
    deltas = pts - center
    dist_sq = np.sum(deltas ** 2, axis=1)
    return float(np.sum(np.exp(-dist_sq / (2 * sigma ** 2))))


def _point_to_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> tuple[float, bool]:
    segment = end - start
    length_sq = float(np.dot(segment, segment))
    if length_sq == 0.0:
        return float(np.linalg.norm(point - start)), False
    t = float(np.dot(point - start, segment) / length_sq)
    t_clamped = np.clip(t, 0.0, 1.0)
    projection = start + t_clamped * segment
    distance = float(np.linalg.norm(point - projection))
    on_segment = 0.0 <= t <= 1.0
    return distance, on_segment


def compute_spatial_context_features(
    df: pd.DataFrame,
    sigma: float = 3.0,
    control_beta: float = 6.0,
    intercept_radius: float = 1.5,
    memory: int = 8,
) -> pd.DataFrame:
    work_df = df.sort_values(["game_episode", "action_id"]).copy()
    positions: dict[tuple[int, int], deque] = defaultdict(lambda: deque(maxlen=memory))
    game_teams: dict[int, set[int]] = defaultdict(set)
    dpi_vals, control_vals, forward_vals, pass_vals = [], [], [], []
    for row in work_df.itertuples():
        if np.isnan(getattr(row, "start_x", np.nan)) or np.isnan(getattr(row, "start_y", np.nan)):
            dpi_vals.append(np.nan)
            control_vals.append(np.nan)
            forward_vals.append(np.nan)
            pass_vals.append(np.nan)
            continue
        center = np.array([row.start_x, row.start_y], dtype=float)
        team_id = getattr(row, "team_id", None)
        game_id = getattr(row, "game_id", None)
        if team_id is None or game_id is None:
            dpi_vals.append(np.nan)
            control_vals.append(np.nan)
            forward_vals.append(np.nan)
            pass_vals.append(np.nan)
            continue
        team_key = (game_id, int(team_id))
        game_teams[game_id].add(int(team_id))
        friendly_positions = list(positions[team_key])
        opponent_positions: list[tuple[float, float]] = []
        for opp_team in game_teams[game_id]:
            if opp_team == int(team_id):
                continue
            opponent_positions.extend(positions[(game_id, opp_team)])
        dpi_vals.append(_gaussian_weights(center, opponent_positions, sigma))
        friendly_weight = _gaussian_weights(center, friendly_positions, control_beta)
        opponent_weight = _gaussian_weights(center, opponent_positions, control_beta)
        Control_margin = friendly_weight - opponent_weight
        control_vals.append(Control_margin)
        forward_friendly = [pos for pos in friendly_positions if pos[0] >= 52.5]
        forward_weight = _gaussian_weights(center, forward_friendly, control_beta)
        forward_ratio = forward_weight / friendly_weight if friendly_weight else 0.0
        forward_vals.append(forward_ratio)
        receivers = [np.array(pos) for pos in friendly_positions if pos != (row.start_x, row.start_y)]
        if not receivers:
            pass_vals.append(0.0)
        else:
            open_count = 0
            for receiver in receivers:
                blocked = False
                for opp in opponent_positions:
                    opp_pt = np.array(opp)
                    distance, on_segment = _point_to_segment_distance(opp_pt, center, receiver)
                    if on_segment and distance < intercept_radius:
                        blocked = True
                        break
                if not blocked:
                    open_count += 1
            pass_vals.append(open_count / len(receivers))
        if not np.isnan(row.start_x):
            positions[team_key].append((row.start_x, row.start_y))
    work_df["dynamic_pressure_index"] = dpi_vals
    work_df["pitch_control_margin"] = control_vals
    work_df["forward_control_ratio"] = forward_vals
    work_df["pass_availability_score"] = pass_vals
    df = df.copy()
    df.loc[work_df.index, [
        "dynamic_pressure_index",
        "pitch_control_margin",
        "forward_control_ratio",
        "pass_availability_score",
    ]] = work_df[[
        "dynamic_pressure_index",
        "pitch_control_margin",
        "forward_control_ratio",
        "pass_availability_score",
    ]]
    return df


def compute_off_ball_energy(df: pd.DataFrame) -> pd.DataFrame:
    if "player_id" not in df.columns:
        return df
    work_df = df.copy()
    work_df["player_key"] = work_df["player_id"].fillna(-1).astype("int64")
    work_df = work_df.sort_values(["player_key", "game_episode", "action_id"])
    grp = work_df.groupby("player_key")
    time_delta = grp["time_seconds"].diff().replace(0, np.nan)
    time_delta = time_delta.bfill().fillna(1.0)
    work_df["player_vx"] = grp["start_x"].diff().fillna(0.0) / time_delta
    work_df["player_vy"] = grp["start_y"].diff().fillna(0.0) / time_delta
    accel_delta = grp["time_seconds"].diff().replace(0, np.nan).fillna(1.0)
    work_df["player_ax"] = grp["player_vx"].diff().fillna(0.0) / accel_delta
    work_df["player_ay"] = grp["player_vy"].diff().fillna(0.0) / accel_delta
    decay = np.exp(-work_df["delta_time_prev"].fillna(0.0) / 5.0)
    work_df["off_ball_energy"] = np.sqrt(work_df["player_ax"] ** 2 + work_df["player_ay"] ** 2) * decay
    df = df.copy()
    for col in ["player_vx", "player_vy", "player_ax", "player_ay", "off_ball_energy"]:
        df.loc[work_df.index, col] = work_df[col]
    df = df.drop(columns="player_key", errors="ignore")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = fill_result_fields(df)
    df = engineer_coordinates(df)
    df = engineer_velocity_and_direction(df)
    df = engineer_relative_goal_features(df)
    df = normalize_coordinates(df)
    df = engineer_temporal(df)
    df = engineer_context_embeddings(df)
    df = compute_attack_momentum(df)
    df = compute_spatial_context_features(df)
    df = compute_off_ball_energy(df)
    return df


def compute_type_weights(df: pd.DataFrame) -> dict[str, float]:
    freq = df["type_name"].value_counts()
    total = len(df)
    weights = (total / (len(freq) * freq)).to_dict()
    return {str(k): float(v) for k, v in weights.items()}


def apply_balancing(df: pd.DataFrame, strategy: BalanceStrategy, pass_ratio: float, random_state: int) -> pd.DataFrame:
    if strategy == "none":
        return df
    if strategy == "type_weight":
        weights = compute_type_weights(df)
        df = df.copy()
        df["type_weight"] = df["type_name"].map(weights).fillna(1.0)
        return df
    # undersample_pass
    pass_mask = df["type_name"].eq("Pass")
    non_pass = df.loc[~pass_mask]
    if non_pass.empty:
        return df
    target_total = int(non_pass.shape[0] / (1 - pass_ratio))
    target_pass = max(1, min(pass_mask.sum(), target_total - non_pass.shape[0]))
    sampled_pass = df.loc[pass_mask].sample(n=target_pass, random_state=random_state)
    balanced = pd.concat([sampled_pass, non_pass]).sample(frac=1, random_state=random_state).reset_index(drop=True)
    return balanced


def save_dataset(df: pd.DataFrame, path: Path, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def resolve_data_path(base_dir: Path, relative_path: str) -> Path:
    rel_path = Path(relative_path)
    if rel_path.is_absolute():
        return rel_path
    return (base_dir / rel_path).resolve()


def load_test_events(meta_path: Path, base_dir: Path, sample_rows: Optional[int]) -> pd.DataFrame:
    if not meta_path.exists():
        raise FileNotFoundError(f"Expected test metadata not found: {meta_path}")
    meta_df = pd.read_csv(meta_path)
    if meta_df.empty:
        return pd.DataFrame()
    frames = []
    for row in meta_df.itertuples(index=False):
        rel_path = getattr(row, "path", None)
        if rel_path is None:
            continue
        event_path = resolve_data_path(base_dir, rel_path)
        if not event_path.exists():
            raise FileNotFoundError(f"Missing test event file: {event_path}")
        event_df = pd.read_csv(event_path)
        if "game_id" not in event_df.columns:
            event_df["game_id"] = getattr(row, "game_id", None)
        if "game_episode" not in event_df.columns:
            event_df["game_episode"] = getattr(row, "game_episode", None)
        frames.append(event_df)
    if not frames:
        return pd.DataFrame()
    test_df = pd.concat(frames, ignore_index=True)
    if sample_rows and len(test_df) > sample_rows:
        test_df = test_df.sample(n=sample_rows, random_state=42).sort_index()
    return test_df


def main() -> None:
    args = parse_args()
    train_path = args.data_dir / "train.csv"
    test_path = args.data_dir / "test.csv"

    train_df = load_csv(train_path, args.sample_rows)
    test_df = (
        load_test_events(test_path, args.data_dir, args.sample_rows)
        if test_path.exists()
        else None
    )

    train_features = engineer_features(train_df)
    train_features = apply_balancing(train_features, args.balance_strategy, args.pass_ratio, args.random_state)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    save_dataset(train_features, output_dir / f"train_features.{args.save_format}", args.save_format)

    if args.balance_strategy == "type_weight":
        weights = compute_type_weights(train_df)
        with open(output_dir / "type_name_weights.json", "w", encoding="utf-8") as fp:
            json.dump(weights, fp, indent=2)

    if test_df is not None:
        test_features = engineer_features(test_df)
        save_dataset(test_features, output_dir / f"test_features.{args.save_format}", args.save_format)

    print("[INFO] Feature engineering complete")


if __name__ == "__main__":
    main()

