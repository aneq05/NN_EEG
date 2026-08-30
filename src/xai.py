from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from lime.lime_tabular import LimeTabularExplainer
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score

from src.evaluation import CLASS_NAMES


def permutation_importance(
    predict_fn,
    x: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    seed: int,
    n_repeats: int = 3,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    baseline = f1_score(y, predict_fn(x), average="macro")
    rows = []
    for j, name in enumerate(feature_names):
        drops = []
        for _ in range(n_repeats):
            x_perm = x.copy()
            x_perm[:, j] = rng.permutation(x_perm[:, j])
            score = f1_score(y, predict_fn(x_perm), average="macro")
            drops.append(baseline - score)
        rows.append({"feature": name, "importance_mean": np.mean(drops), "importance_std": np.std(drops)})
    return pd.DataFrame(rows).sort_values("importance_mean", ascending=False)


def segment_importance(feature_importance: pd.DataFrame, segment_size: int = 30) -> pd.DataFrame:
    rows = []
    for _, row in feature_importance.iterrows():
        idx = int(str(row["feature"]).lstrip("Xx"))
        start = ((idx - 1) // segment_size) * segment_size + 1
        end = min((((idx - 1) // segment_size) + 1) * segment_size, 178)
        rows.append({**row.to_dict(), "segment": f"X{start}-X{end}"})
    return (
        pd.DataFrame(rows)
        .groupby("segment", as_index=False)["importance_mean"]
        .sum()
        .sort_values("importance_mean", ascending=False)
    )


def save_importance_plot(df: pd.DataFrame, output_path: Path, title: str) -> None:
    top = df.head(20).iloc[::-1]
    plt.figure(figsize=(8, 6))
    plt.barh(top.iloc[:, 0], top["importance_mean"])
    plt.xlabel("Macro F1 drop after permutation")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def build_lime_explainer(x_train: np.ndarray, feature_names: list[str], seed: int) -> LimeTabularExplainer:
    return LimeTabularExplainer(
        training_data=x_train,
        feature_names=feature_names,
        class_names=CLASS_NAMES,
        mode="classification",
        discretize_continuous=True,
        random_state=seed,
    )


def map_lime_feature(text: str, feature_names: list[str]) -> str:
    for name in feature_names:
        if name in text:
            return name
    return text


def global_lime(
    explainer: LimeTabularExplainer,
    predict_proba_fn,
    x: np.ndarray,
    feature_names: list[str],
    seed: int,
    n_instances: int = 120,
) -> pd.DataFrame:
    totals = defaultdict(float)
    counts = defaultdict(int)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(x), size=min(n_instances, len(x)), replace=False)
    for idx in indices:
        exp = explainer.explain_instance(x[idx], predict_proba_fn, num_features=12, num_samples=3000)
        for feat, weight in exp.as_list():
            clean = map_lime_feature(feat, feature_names)
            totals[clean] += abs(weight)
            counts[clean] += 1
    rows = [{"feature": key, "mean_abs_lime_weight": totals[key] / counts[key]} for key in totals]
    return pd.DataFrame(rows).sort_values("mean_abs_lime_weight", ascending=False)


def save_lime_plot(df: pd.DataFrame, figures_dir: Path) -> None:
    top = df.head(20).iloc[::-1]
    plt.figure(figsize=(8, 6))
    plt.barh(top["feature"], top["mean_abs_lime_weight"])
    plt.xlabel("Mean absolute LIME weight")
    plt.title("Global LIME importance")
    plt.tight_layout()
    plt.savefig(Path(figures_dir) / "lime_global.png", dpi=180)
    plt.close()


def save_local_lime(
    explainer: LimeTabularExplainer,
    predict_proba_fn,
    x_test: np.ndarray,
    cases: dict[str, pd.DataFrame],
    lime_dir: Path,
) -> None:
    lime_dir = Path(lime_dir)
    lime_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for group_name, group in cases.items():
        for _, row in group.head(4).iterrows():
            idx = int(row["index"])
            exp = explainer.explain_instance(x_test[idx], predict_proba_fn, num_features=12, num_samples=3000)
            html_path = lime_dir / f"{group_name}_{idx}.html"
            exp.save_to_file(str(html_path))
            for feat, weight in exp.as_list():
                summaries.append(
                    {
                        "group": group_name,
                        "test_index": idx,
                        "feature": map_lime_feature(feat, explainer.feature_names),
                        "weight": weight,
                    }
                )
    pd.DataFrame(summaries).to_csv(lime_dir / "local_lime_summary.csv", index=False)
