import json

import pandas as pd
import torch

from src.train import best_validation_score, hpo_cache_config, hpo_cache_is_valid


def test_hpo_cache_requires_matching_experiment_config(tmp_path):
    config_path = tmp_path / "hpo_config.json"
    trials_path = tmp_path / "hyperparameter_search.csv"
    expected = hpo_cache_config(n_trials=2, max_epochs=3, seed=42)

    config_path.write_text(json.dumps(expected), encoding="utf-8")
    pd.DataFrame({"number": [0, 1], "value": [0.1, 0.2]}).to_csv(trials_path, index=False)

    assert hpo_cache_is_valid(config_path, trials_path, expected)
    assert not hpo_cache_is_valid(config_path, trials_path, hpo_cache_config(n_trials=3, max_epochs=3, seed=42))


def test_hpo_uses_best_validation_score():
    class FakeEarlyStopping:
        best_score = torch.tensor(0.765)

    assert best_validation_score(FakeEarlyStopping()) == torch.tensor(0.765).item()
