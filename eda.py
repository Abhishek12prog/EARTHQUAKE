from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import PLOTS_DIR, PROCESSED_DATA_PATH


def run_eda(input_path=PROCESSED_DATA_PATH):
    sns.set_theme(style="whitegrid", palette="viridis")
    df = pd.read_csv(input_path)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).copy()
    plots_dir = Path(PLOTS_DIR)
    plots_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    sns.histplot(df["magnitude"], bins=40, kde=True, color="teal")
    plt.title("Magnitude Distribution")
    plt.xlabel("Magnitude")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(plots_dir / "magnitude_distribution.png", dpi=200, bbox_inches="tight")
    plt.close()

    monthly_counts = (
        df.set_index("time")
        .resample("ME")
        .size()
        .reset_index(name="earthquake_count")
    )
    plt.figure(figsize=(14, 6))
    plt.plot(monthly_counts["time"], monthly_counts["earthquake_count"], color="darkorange")
    plt.title("Earthquake Frequency Over Time")
    plt.xlabel("Time")
    plt.ylabel("Number of Earthquakes")
    plt.tight_layout()
    plt.savefig(plots_dir / "earthquake_frequency_over_time.png", dpi=200, bbox_inches="tight")
    plt.close()

    sample_df = df.sample(min(10000, len(df)), random_state=42)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=sample_df,
        x="depth",
        y="magnitude",
        hue="risk_level",
        alpha=0.7,
    )
    plt.title("Depth vs Magnitude")
    plt.xlabel("Depth")
    plt.ylabel("Magnitude")
    plt.tight_layout()
    plt.savefig(plots_dir / "depth_vs_magnitude.png", dpi=200, bbox_inches="tight")
    plt.close()

    numeric_cols = [
        "latitude",
        "longitude",
        "depth",
        "magnitude",
        "year",
        "month",
        "day",
        "hour",
        "day_of_week",
    ]
    corr_matrix = df[numeric_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(plots_dir / "correlation_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved EDA plots to {plots_dir.resolve()}")


if __name__ == "__main__":
    run_eda()
