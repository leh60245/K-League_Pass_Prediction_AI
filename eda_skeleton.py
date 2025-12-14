"""Baseline EDA scaffold for the K-League final-pass prediction challenge.

Run this script after placing match_info.csv, train.csv, and test.csv inside a data/
folder (or pass explicit paths). It writes lightweight profiling tables and plots
into artifacts/eda for quick inspection.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

try:  # Plotly is optional; continue without it if unavailable
    import plotly.express as px
except ImportError:  # pragma: no cover - runtime fallback
    px = None

sns.set_theme(style="whitegrid")
ARTIFACT_ROOT = Path("artifacts") / "eda"


@dataclass
class DatasetBundle:
    match_info: Optional[Path]
    train: Optional[Path]
    test: Optional[Path]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate baseline EDA artifacts")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Folder containing CSV files")
    parser.add_argument("--match-info", type=Path, default=None, help="Optional explicit match_info.csv path")
    parser.add_argument("--train", type=Path, default=None, help="Optional explicit train.csv path")
    parser.add_argument("--test", type=Path, default=None, help="Optional explicit test.csv path")
    parser.add_argument("--sample-rows", type=int, default=None, help="Randomly sample rows per file for quicker runs")
    parser.add_argument("--output-dir", type=Path, default=ARTIFACT_ROOT, help="Directory to store EDA outputs")
    parser.add_argument("--config-out", type=Path, default=ARTIFACT_ROOT / "run_config.json", help="Path to dump resolved run metadata")
    return parser


def resolve_bundle(args: argparse.Namespace) -> DatasetBundle:
    base = args.data_dir
    return DatasetBundle(
        match_info=args.match_info or base / "match_info.csv",
        train=args.train or base / "train.csv",
        test=args.test or base / "test.csv",
    )


def ensure_output_dir(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_run_config(args: argparse.Namespace, bundle: DatasetBundle) -> None:
    payload = {
        "args": {
            "data_dir": str(args.data_dir),
            "sample_rows": args.sample_rows,
            "output_dir": str(args.output_dir),
        },
        "resolved_paths": {
            "match_info": str(bundle.match_info) if bundle.match_info else None,
            "train": str(bundle.train) if bundle.train else None,
            "test": str(bundle.test) if bundle.test else None,
        },
    }
    args.config_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.config_out, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)
    print(f"[INFO] Run configuration saved to {args.config_out}")


def read_csv(path: Optional[Path], parse_dates: Optional[list[str]] = None, sample_rows: Optional[int] = None) -> Optional[pd.DataFrame]:
    if path is None:
        return None
    if not path.exists():
        print(f"[WARN] Missing file: {path}")
        return None
    df = pd.read_csv(path, parse_dates=parse_dates)
    if sample_rows and len(df) > sample_rows:
        df = df.sample(n=sample_rows, random_state=42).sort_index()
    print(f"[INFO] Loaded {path.name} -> shape {df.shape}")
    return df


def profile_dataframe(df: pd.DataFrame, name: str, output_dir: Path) -> None:
    profile = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "non_null": df.notna().sum(),
            "missing_pct": (df.isna().mean() * 100).round(2),
            "unique": df.nunique(dropna=True),
        }
    )
    profile.index.name = "column"
    profile.to_csv(output_dir / f"{name}_column_profile.csv")
    summary = df.describe(include="all").transpose()
    summary.to_csv(output_dir / f"{name}_describe.csv")
    print(f"[INFO] Profiling tables stored for {name}")


def summarize_match_info(df: pd.DataFrame, output_dir: Path) -> None:
    summary = (
        df.groupby(["competition_name", "season_name"], dropna=False)
        .agg(
            matches=("game_id", "nunique"),
            home_teams=("home_team_id", "nunique"),
            away_teams=("away_team_id", "nunique"),
            avg_home_goals=("home_score", "mean"),
            avg_away_goals=("away_score", "mean"),
        )
        .reset_index()
    )
    summary["unique_teams_est"] = summary[["home_teams", "away_teams"]].max(axis=1)
    summary.to_csv(output_dir / "match_info_competition_summary.csv", index=False)
    print("[INFO] Match metadata summary exported")


def summarize_event_catalog(train_df: pd.DataFrame, output_dir: Path) -> None:
    type_counts = train_df["type_name"].value_counts(dropna=False).reset_index()
    type_counts.columns = ["type_name", "count"]
    type_counts.to_csv(output_dir / "event_type_counts.csv", index=False)
    result_counts = (
        train_df.groupby(["type_name", "result_name"], dropna=False).size().reset_index(name="count")
    )
    result_counts.to_csv(output_dir / "event_type_result_counts.csv", index=False)
    print("[INFO] Event catalog summaries written")


def summarize_episode_level(train_df: pd.DataFrame, output_dir: Path) -> None:
    aggregations = {
        "action_id": "count",
        "time_seconds": lambda s: float(s.max() - s.min()) if len(s) else 0.0,
        "game_id": "nunique",
        "period_id": ["min", "max"],
    }
    grouped = train_df.groupby("game_episode").agg(aggregations)
    grouped.columns = ["event_count", "duration_seconds", "game_count", "period_min", "period_max"]
    grouped.to_csv(output_dir / "episode_level_stats.csv")
    print("[INFO] Episode level stats exported")


def plot_top_events(train_df: pd.DataFrame, output_dir: Path) -> None:
    top_events = train_df["type_name"].value_counts().head(20)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=top_events.values, y=top_events.index, ax=ax, orient="h", color="steelblue")
    ax.set_title("Top 20 event types")
    ax.set_xlabel("count")
    ax.set_ylabel("type_name")
    fig.tight_layout()
    fig.savefig(output_dir / "top_event_types.png", dpi=200)
    plt.close(fig)
    print("[INFO] Top event plot saved")


def plot_coordinate_heatmap(train_df: pd.DataFrame, output_dir: Path, coord_prefix: str) -> None:
    x_col = f"{coord_prefix}_x"
    y_col = f"{coord_prefix}_y"
    if x_col not in train_df.columns or y_col not in train_df.columns:
        print(f"[WARN] Missing coordinate columns for {coord_prefix}")
        return
    coords = train_df[[x_col, y_col]].dropna()
    if coords.empty:
        print(f"[WARN] No coordinates to plot for {coord_prefix}")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.kdeplot(data=coords, x=x_col, y=y_col, fill=True, cmap="viridis", levels=80, thresh=0.05, ax=ax)
    ax.set_title(f"{coord_prefix.title()} coordinate density")
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 68)
    ax.set_aspect(105 / 68)
    fig.tight_layout()
    fig.savefig(output_dir / f"{coord_prefix}_coordinate_density.png", dpi=200)
    plt.close(fig)
    if px is not None:
        fig_px = px.density_heatmap(
            coords,
            x=x_col,
            y=y_col,
            nbinsx=60,
            nbinsy=40,
            title=f"Interactive {coord_prefix} coordinate density",
            range_x=[0, 105],
            range_y=[0, 68],
        )
        fig_px.write_html(output_dir / f"{coord_prefix}_coordinate_density.html")
    print(f"[INFO] Coordinate heatmap stored for {coord_prefix}")


def plot_episode_tempo(train_df: pd.DataFrame, output_dir: Path) -> None:
    sort_cols = ["game_id", "period_id", "time_seconds", "action_id"]
    df = train_df.sort_values(sort_cols)
    df["period_elapsed"] = df.groupby(["game_id", "period_id"])["time_seconds"].transform(lambda s: s - s.min())
    sample_games = df["game_id"].dropna().unique()[:3]
    subset = df[df["game_id"].isin(sample_games)]
    if subset.empty:
        print("[WARN] Not enough events for tempo plot")
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=subset, x="period_elapsed", y="action_id", hue="game_id", style="period_id", ax=ax)
    ax.set_title("Sample action tempo by game")
    ax.set_xlabel("seconds since period start")
    ax.set_ylabel("action_id")
    fig.tight_layout()
    fig.savefig(output_dir / "sample_action_tempo.png", dpi=200)
    plt.close(fig)
    print("[INFO] Tempo plot saved")


def run_eda(args: argparse.Namespace) -> None:
    bundle = resolve_bundle(args)
    ensure_output_dir(args.output_dir)
    save_run_config(args, bundle)

    match_info_df = read_csv(bundle.match_info, parse_dates=["game_date"], sample_rows=args.sample_rows)
    train_df = read_csv(bundle.train, sample_rows=args.sample_rows)
    test_df = read_csv(bundle.test, sample_rows=args.sample_rows)

    if match_info_df is not None:
        profile_dataframe(match_info_df, "match_info", args.output_dir)
        summarize_match_info(match_info_df, args.output_dir)
    if train_df is not None:
        profile_dataframe(train_df, "train", args.output_dir)
        summarize_event_catalog(train_df, args.output_dir)
        summarize_episode_level(train_df, args.output_dir)
        plot_top_events(train_df, args.output_dir)
        plot_coordinate_heatmap(train_df, args.output_dir, "start")
        plot_coordinate_heatmap(train_df, args.output_dir, "end")
        plot_episode_tempo(train_df, args.output_dir)
    if test_df is not None:
        profile_dataframe(test_df, "test", args.output_dir)

    print("[INFO] EDA scaffold finished")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_eda(args)


if __name__ == "__main__":
    main()

