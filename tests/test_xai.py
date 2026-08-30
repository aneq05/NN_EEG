import numpy as np
import pandas as pd

from src.xai import global_lime, map_lime_feature, save_local_lime


def test_map_lime_feature_uses_exact_feature_token():
    features = [f"X{i}" for i in range(1, 179)]

    assert map_lime_feature("X10 <= 0.4", features) == "X10"
    assert map_lime_feature("-0.2 < X178 <= 1.1", features) == "X178"
    assert map_lime_feature("no matching feature", features) == "no matching feature"


class FakeExplanation:
    def __init__(self, expected_label: int):
        self.expected_label = expected_label

    def as_list(self, label=None):
        assert label == self.expected_label
        return [("X10 <= 0.4", 0.5), ("-0.2 < X178 <= 1.1", -0.25)]

    def save_to_file(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("fake lime html")


class FakeExplainer:
    feature_names = [f"X{i}" for i in range(1, 179)]

    def __init__(self):
        self.labels_seen = []

    def explain_instance(self, instance, predict_proba_fn, labels, num_features, num_samples):
        assert num_features == 12
        assert num_samples == 3000
        self.labels_seen.append(labels)
        return FakeExplanation(labels[0])


def fake_predict_proba(x):
    probs = np.tile(np.array([[0.1, 0.7, 0.2]]), (len(x), 1))
    return probs


def test_global_lime_explains_predicted_class_with_exact_features():
    explainer = FakeExplainer()
    result = global_lime(
        explainer,
        fake_predict_proba,
        np.zeros((2, 178), dtype="float32"),
        explainer.feature_names,
        seed=42,
        n_instances=2,
    )

    assert explainer.labels_seen == [(1,), (1,)]
    assert result["feature"].tolist() == ["X10", "X178"]


def test_save_local_lime_records_class_context_and_removes_stale_html(tmp_path):
    explainer = FakeExplainer()
    stale = tmp_path / "stale.html"
    stale.write_text("old", encoding="utf-8")
    cases = {"most_confident_correct": pd.DataFrame({"index": [0]})}

    save_local_lime(
        explainer,
        fake_predict_proba,
        np.zeros((1, 178), dtype="float32"),
        np.array([2]),
        cases,
        tmp_path,
    )

    summary = pd.read_csv(tmp_path / "local_lime_summary.csv")
    assert not stale.exists()
    assert (tmp_path / "most_confident_correct_0.html").exists()
    assert summary[["sample_index", "predicted_class", "true_class"]].iloc[0].tolist() == [0, 1, 2]
    assert summary["feature"].tolist() == ["X10", "X178"]
