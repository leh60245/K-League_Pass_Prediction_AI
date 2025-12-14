"""Generate feature distribution plots and narrative summaries for engineered soccer data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Mapping

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DEFAULT_INPUT = Path("artifacts") / "features" / "train_features.parquet"
DEFAULT_OUTPUT = Path("artifacts") / "reports"
NUMERIC_FEATURES = [
    "start_x",
    "start_y",
    "end_x",
    "dist_to_goal",
    "dist_to_goal_norm",
    "delta_time_prev",
    "speed",
    "speed_clipped",
    "attack_progress_momentum",
    "dynamic_pressure_index",
    "pitch_control_margin",
    "forward_control_ratio",
    "pass_availability_score",
    "off_ball_energy",
]
CATEGORICAL_FEATURES = [
    "type_filled",
    "result_filled",
    "event_context",
    "prev_event_context",
    "start_quadrant",
    "end_quadrant",
    "type_group",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot feature distributions and write textual summaries")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to engineered dataset (parquet/csv)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Directory to store plots and summaries")
    parser.add_argument("--sample-rows", type=int, default=None, help="Optional row cap for quick previews")
    parser.add_argument("--format", choices=["parquet", "csv", "auto"], default="auto", help="Input format override if extension missing")
    return parser.parse_args()


def load_dataset(path: Path, fmt: str, sample_rows: int | None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    fmt = fmt if fmt != "auto" else path.suffix.lstrip(".")
    if fmt == "parquet":
        df = pd.read_parquet(path)
    elif fmt == "csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported format: {fmt}")
    if sample_rows and len(df) > sample_rows:
        df = df.sample(n=sample_rows, random_state=42).sort_index()
    return df.reset_index(drop=True)


def ensure_output_dir(output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    (output / "plots").mkdir(exist_ok=True)
    return output


def summarize_numeric(series: pd.Series) -> Mapping[str, float]:
    desc = series.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
    summary = {k: float(desc.get(k, 0.0)) for k in desc.index}
    summary["skew"] = float(series.skew(skipna=True))
    summary["missing_pct"] = float(series.isna().mean() * 100)
    return summary


def summarize_categorical(series: pd.Series, top_k: int = 10) -> Mapping[str, object]:
    counts = series.value_counts(dropna=False).head(top_k)
    top = [{"value": str(idx), "count": int(cnt)} for idx, cnt in counts.items()]
    missing_pct = float(series.isna().mean() * 100)
    coverage = float(counts.sum() / len(series) * 100) if len(series) else 0.0
    return {"top": top, "missing_pct": missing_pct, "coverage_pct": coverage}


def write_markdown_report(output: Path, numeric_stats: Mapping[str, Mapping[str, float]], categorical_stats: Mapping[str, Mapping[str, object]]) -> None:
    lines: list[str] = ["# Feature Distribution Report", ""]
    if numeric_stats:
        lines.append("## Numeric Features")
        for name, stats in numeric_stats.items():
            lines.extend([
                f"### {name}",
                f"- count: {stats.get('count', 0):,.0f}",
                f"- mean ± std: {stats.get('mean', 0.0):.3f} ± {stats.get('std', 0.0):.3f}",
                f"- median (p5-p95): {stats.get('50%', 0.0):.3f} ({stats.get('5%', 0.0):.3f}–{stats.get('95%', 0.0):.3f})",
                f"- min / max: {stats.get('min', 0.0):.3f} / {stats.get('max', 0.0):.3f}",
                f"- skewness: {stats.get('skew', 0.0):.3f}",
                f"- missing: {stats.get('missing_pct', 0.0):.2f}%",
                "",
            ])
    if categorical_stats:
        lines.append("## Categorical Features")
        for name, stats in categorical_stats.items():
            lines.append(f"### {name}")
            lines.append(f"- missing: {stats.get('missing_pct', 0.0):.2f}%")
            lines.append(f"- coverage (top bucket share): {stats.get('coverage_pct', 0.0):.2f}%")
            lines.append("- top categories:")
            for entry in stats.get("top", []):
                lines.append(f"  - {entry['value']}: {entry['count']:,}")
            lines.append("")
    report_path = output / "feature_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")


def save_json_summary(output: Path, numeric_stats: Mapping[str, Mapping[str, float]], categorical_stats: Mapping[str, Mapping[str, object]]) -> None:
    payload = {"numeric": numeric_stats, "categorical": categorical_stats}
    with open(output / "feature_report.json", "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)


def plot_numeric_features(df: pd.DataFrame, columns: Iterable[str], output: Path) -> None:
    for col in columns:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if series.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(series, bins=40, kde=True, ax=ax, color="steelblue")
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("count")
        fig.tight_layout()
        fig.savefig(output / "plots" / f"{col}_hist.png", dpi=200)
        plt.close(fig)


def plot_categorical_features(df: pd.DataFrame, columns: Iterable[str], output: Path, top_k: int = 15) -> None:
    for col in columns:
        if col not in df.columns:
            continue
        counts = df[col].value_counts().head(top_k)
        if counts.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 4 + 0.2 * len(counts)))
        sns.barplot(x=counts.values, y=counts.index.astype(str), ax=ax, color="darkorange")
        ax.set_title(f"Top {len(counts)} categories for {col}")
        ax.set_xlabel("count")
        ax.set_ylabel(col)
        fig.tight_layout()
        fig.savefig(output / "plots" / f"{col}_bar.png", dpi=200)
        plt.close(fig)


def run_report(args: argparse.Namespace) -> None:
    df = load_dataset(args.input, args.format, args.sample_rows)
    output_dir = ensure_output_dir(args.output)

    numeric_stats = {}
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            numeric_stats[col] = summarize_numeric(df[col])
    categorical_stats = {}
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            categorical_stats[col] = summarize_categorical(df[col])

    plot_numeric_features(df, numeric_stats.keys(), output_dir)
    plot_categorical_features(df, categorical_stats.keys(), output_dir)
    write_markdown_report(output_dir, numeric_stats, categorical_stats)
    save_json_summary(output_dir, numeric_stats, categorical_stats)
    print(f"[INFO] Feature report saved to {output_dir}")


def main() -> None:
    args = parse_args()
    run_report(args)


if __name__ == "__main__":
    main()

