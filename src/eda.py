from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

from src.preprocessing import feature_columns


def run_eda(df: pd.DataFrame, splits: dict, cfg: dict) -> None:
    figures_dir = Path(cfg["paths"]["figures_dir"])
    tables_dir = Path(cfg["paths"]["tables_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    features = feature_columns(df)

    profile = pd.DataFrame(
        {
            "n_rows": [len(df)],
            "n_features": [len(features)],
            "missing_values": [int(df.isna().sum().sum())],
            "duplicates": [int(df.duplicated().sum())],
        }
    )
    profile.to_csv(tables_dir / "data_cleaning_summary.csv", index=False)
    df["y"].value_counts().sort_index().rename_axis("class").reset_index(name="count").to_csv(
        tables_dir / "class_distribution.csv", index=False
    )
    df[features].describe().T.to_csv(tables_dir / "feature_describe.csv")

    plt.figure(figsize=(7, 4))
    order = sorted(df["y"].unique())
    sns.countplot(data=df, x="y", order=order)
    plt.title("Class distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(figures_dir / "class_distribution.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 6))
    for cls in order:
        sample = df[df["y"] == cls].iloc[0][features].to_numpy()
        plt.plot(sample, label=f"class {cls}", linewidth=1)
    plt.title("Example EEG signals by class")
    plt.xlabel("Time point")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "example_signals.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 6))
    for cls in order:
        mean_signal = df[df["y"] == cls][features].mean(axis=0).to_numpy()
        plt.plot(mean_signal, label=f"class {cls}", linewidth=1.5)
    plt.title("Mean EEG signal by class")
    plt.xlabel("Time point")
    plt.ylabel("Mean amplitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "mean_signals.png", dpi=180)
    plt.close()

    eda = df[["y"]].copy()
    eda["signal_mean"] = df[features].mean(axis=1)
    eda["signal_std"] = df[features].std(axis=1)
    eda["signal_min"] = df[features].min(axis=1)
    eda["signal_max"] = df[features].max(axis=1)
    eda["signal_energy"] = (df[features] ** 2).sum(axis=1)
    eda.describe().T.to_csv(tables_dir / "signal_level_features_describe.csv")

    plt.figure(figsize=(7, 5))
    sns.boxplot(data=eda, x="y", y="signal_energy")
    plt.title("Signal energy by class")
    plt.xlabel("Class")
    plt.ylabel("Energy")
    plt.tight_layout()
    plt.savefig(figures_dir / "signal_energy_by_class.png", dpi=180)
    plt.close()

    pca = PCA(n_components=2, random_state=cfg["project"]["seed"])
    x_pca = pca.fit_transform(splits["x_train"])
    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(x_pca[:, 0], x_pca[:, 1], c=splits["y_train"], s=8, alpha=0.55, cmap="tab10")
    plt.title("PCA of scaled training samples")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(scatter, label="Encoded class")
    plt.tight_layout()
    plt.savefig(figures_dir / "pca_train.png", dpi=180)
    plt.close()
