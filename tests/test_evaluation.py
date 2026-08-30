import numpy as np

from src.evaluation import metrics_dict, uncertainty_table


def test_metrics_dict_contains_core_multiclass_metrics():
    y_true = np.array([0, 1, 2, 1])
    probs = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.7, 0.2],
            [0.2, 0.1, 0.7],
            [0.2, 0.6, 0.2],
        ]
    )

    metrics = metrics_dict(y_true, probs)

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert "macro_ovr_auroc" in metrics


def test_uncertainty_table_marks_correct_predictions():
    y_true = np.array([0, 1])
    probs = np.array([[0.7, 0.3], [0.6, 0.4]])

    table = uncertainty_table(y_true, probs)

    assert table["correct"].tolist() == [True, False]
    assert {"max_probability", "entropy", "margin_top1_top2"}.issubset(table.columns)

