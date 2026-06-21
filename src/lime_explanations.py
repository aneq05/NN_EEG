from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer


def build_explainer(x_train: np.ndarray, feature_names: list[str], class_names: list[str]) -> LimeTabularExplainer:
    return LimeTabularExplainer(
        training_data=x_train,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification",
        discretize_continuous=True,
        random_state=42,
    )


def global_lime(
    explainer: LimeTabularExplainer,
    predict_proba_fn,
    x: np.ndarray,
    feature_names: list[str],
    n_instances: int,
    num_features: int,
    num_samples: int,
) -> pd.DataFrame:
    totals = defaultdict(float)
    counts = defaultdict(int)
    n = min(n_instances, len(x))
    rng = np.random.default_rng(42)
    indices = rng.choice(len(x), size=n, replace=False)
    for idx in indices:
        exp = explainer.explain_instance(
            x[idx],
            predict_proba_fn,
            num_features=num_features,
            num_samples=num_samples,
        )
        for feat, weight in exp.as_list():
            clean = _map_lime_feature(feat, feature_names)
            totals[clean] += abs(weight)
            counts[clean] += 1
    rows = [{"feature": k, "mean_abs_lime_weight": totals[k] / counts[k]} for k in totals]
    return pd.DataFrame(rows).sort_values("mean_abs_lime_weight", ascending=False)


def save_local_lime(
    explainer: LimeTabularExplainer,
    predict_proba_fn,
    x_test: np.ndarray,
    cases: dict[str, pd.DataFrame],
    output_dir: str | Path,
    num_features: int,
    num_samples: int,
    cases_per_group: int,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for group_name, group in cases.items():
        for _, row in group.head(cases_per_group).iterrows():
            idx = int(row["index"])
            exp = explainer.explain_instance(
                x_test[idx],
                predict_proba_fn,
                num_features=num_features,
                num_samples=num_samples,
            )
            html_path = output_dir / f"{group_name}_{idx}.html"
            exp.save_to_file(str(html_path))
            for feat, weight in exp.as_list():
                summaries.append(
                    {
                        "group": group_name,
                        "test_index": idx,
                        "feature": _map_lime_feature(feat, explainer.feature_names),
                        "weight": weight,
                    }
                )
    pd.DataFrame(summaries).to_csv(output_dir / "local_lime_summary.csv", index=False)


def save_lime_plot(df: pd.DataFrame, output_path: str | Path) -> None:
    top = df.head(20).iloc[::-1]
    plt.figure(figsize=(8, 6))
    plt.barh(top["feature"], top["mean_abs_lime_weight"])
    plt.xlabel("Mean absolute LIME weight")
    plt.title("Global LIME importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _map_lime_feature(text: str, feature_names: list[str]) -> str:
    for name in feature_names:
        if name in text:
            return name
    return text
