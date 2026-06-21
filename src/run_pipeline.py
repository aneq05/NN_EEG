from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import lightning as L
import numpy as np
import optuna
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from src.calibrate import TemperatureScaler, predict_logits, softmax
from src.config import ensure_dirs, load_config
from src.data_module import EEGDataModule
from src.download_data import download_dataset
from src.eda import run_eda
from src.evaluate import save_evaluation
from src.lime_explanations import build_explainer, global_lime, save_lime_plot, save_local_lime
from src.model import EEGMLP, ModelConfig
from src.permutation_importance import permutation_importance, save_importance, segment_importance
from src.preprocessing import clean_dataframe, load_raw_data, split_and_scale
from src.uncertainty import save_uncertainty_cases, uncertainty_table


def set_seed(seed: int) -> None:
    L.seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision("medium")


def make_model(cfg: dict, overrides: dict | None = None) -> EEGMLP:
    overrides = overrides or {}
    model_cfg = ModelConfig(
        input_dim=cfg["model"]["input_dim"],
        num_classes=cfg["model"]["num_classes"],
        hidden_dims=overrides.get("hidden_dims", cfg["model"]["hidden_dims"]),
        dropout=overrides.get("dropout", cfg["model"]["dropout"]),
        learning_rate=overrides.get("learning_rate", cfg["training"]["learning_rate"]),
        weight_decay=overrides.get("weight_decay", cfg["training"]["weight_decay"]),
    )
    return EEGMLP(model_cfg)


def trainer_for(
    cfg: dict,
    callbacks: list | None = None,
    quiet: bool = False,
    checkpointing: bool = False,
) -> L.Trainer:
    return L.Trainer(
        max_epochs=cfg["training"]["max_epochs"],
        accelerator=cfg["training"]["accelerator"],
        devices=1,
        callbacks=callbacks or [],
        logger=False,
        enable_checkpointing=checkpointing,
        enable_progress_bar=not quiet,
        deterministic=True,
    )


def optimize_hyperparameters(cfg: dict, data: EEGDataModule) -> tuple[dict, pd.DataFrame]:
    best_path = Path(cfg["paths"]["models_dir"]) / "best_hyperparameters.json"
    trials_path = Path(cfg["paths"]["tables_dir"]) / "hyperparameter_search.csv"
    if best_path.exists() and trials_path.exists():
        return json.loads(best_path.read_text(encoding="utf-8")), pd.read_csv(trials_path)

    if not cfg["optuna"]["enabled"]:
        return {}, pd.DataFrame()

    def objective(trial: optuna.Trial) -> float:
        architectures = {
            "small": [128, 64],
            "medium": [256, 128, 64],
            "large": [512, 256, 128],
        }
        arch_name = trial.suggest_categorical("architecture", list(architectures))
        overrides = {
            "hidden_dims": architectures[arch_name],
            "dropout": trial.suggest_float("dropout", 0.1, 0.5, step=0.1),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        }
        model = make_model(cfg, overrides)
        early = EarlyStopping(monitor="val_macro_f1", mode="max", patience=5)
        trainer = trainer_for(cfg, callbacks=[early], quiet=True, checkpointing=False)
        trainer.fit(model, datamodule=data)
        return float(trainer.callback_metrics["val_macro_f1"].detach().cpu().item())

    study = optuna.create_study(direction="maximize")
    study.optimize(
        objective,
        n_trials=cfg["optuna"]["n_trials"],
        timeout=cfg["optuna"]["timeout_seconds"],
        show_progress_bar=False,
    )
    trials = study.trials_dataframe()
    trials.to_csv(Path(cfg["paths"]["tables_dir"]) / "hyperparameter_search.csv", index=False)
    best = study.best_params
    best["hidden_dims"] = {"small": [128, 64], "medium": [256, 128, 64], "large": [512, 256, 128]}[
        best.pop("architecture")
    ]
    return best, trials


def train_final_model(cfg: dict, data: EEGDataModule, best_params: dict) -> EEGMLP:
    checkpoint = ModelCheckpoint(
        dirpath=cfg["paths"]["models_dir"],
        filename="best_eeg_mlp",
        monitor="val_macro_f1",
        mode="max",
        save_top_k=1,
    )
    early = EarlyStopping(monitor="val_macro_f1", mode="max", patience=cfg["training"]["patience"])
    model = make_model(cfg, best_params)
    trainer = trainer_for(cfg, callbacks=[checkpoint, early], quiet=False, checkpointing=True)
    trainer.fit(model, datamodule=data)
    if checkpoint.best_model_path:
        model = EEGMLP.load_from_checkpoint(checkpoint.best_model_path, weights_only=False)
    torch.save(model.state_dict(), Path(cfg["paths"]["models_dir"]) / "eeg_mlp_state_dict.pt")
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--skip-lime", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    set_seed(cfg["project"]["seed"])
    download_dataset(args.config)

    df_raw = load_raw_data(cfg["paths"]["raw_data"])
    df = clean_dataframe(df_raw, cfg["data"]["id_columns"], cfg["data"]["remove_duplicates"])
    splits = split_and_scale(
        df,
        cfg["paths"]["processed_dir"],
        cfg["data"]["test_size"],
        cfg["data"]["val_size"],
        cfg["data"]["random_state"],
        cfg["data"]["outlier_clip_quantiles"],
    )
    joblib.dump(splits["feature_names"], Path(cfg["paths"]["processed_dir"]) / "feature_names.joblib")
    run_eda(df, splits, cfg)

    data = EEGDataModule(
        splits["x_train"],
        splits["y_train"],
        splits["x_val"],
        splits["y_val"],
        splits["x_test"],
        splits["y_test"],
        cfg["training"]["batch_size"],
    )
    best_params, _ = optimize_hyperparameters(cfg, data)
    Path(cfg["paths"]["models_dir"]).mkdir(parents=True, exist_ok=True)
    (Path(cfg["paths"]["models_dir"]) / "best_hyperparameters.json").write_text(
        json.dumps(best_params, indent=2), encoding="utf-8"
    )
    model = train_final_model(cfg, data, best_params)

    val_logits = predict_logits(model, splits["x_val"])
    test_logits = predict_logits(model, splits["x_test"])
    calibrator = TemperatureScaler().fit(val_logits, splits["y_val"])
    joblib.dump(calibrator, Path(cfg["paths"]["models_dir"]) / "temperature_scaler.joblib")

    probs_before = softmax(test_logits)
    probs_after = softmax(calibrator.transform_logits(test_logits))
    label_names = [cfg["labels"][i] for i in sorted(cfg["labels"])]
    save_evaluation(
        splits["y_test"],
        probs_before,
        probs_after,
        label_names,
        cfg["paths"]["metrics_dir"],
        cfg["paths"]["figures_dir"],
    )

    predictions = pd.DataFrame(probs_after, columns=[f"prob_{name}" for name in label_names])
    predictions.insert(0, "pred", probs_after.argmax(axis=1))
    predictions.insert(0, "true", splits["y_test"])
    predictions.to_csv(Path(cfg["paths"]["tables_dir"]) / "test_predictions_calibrated.csv", index=False)

    uncertainty = uncertainty_table(splits["y_test"], probs_after)
    cases = save_uncertainty_cases(uncertainty, cfg["paths"]["tables_dir"])

    def predict_fn(x: np.ndarray) -> np.ndarray:
        logits = predict_logits(model, x)
        return softmax(calibrator.transform_logits(logits)).argmax(axis=1)

    def predict_proba_fn(x: np.ndarray) -> np.ndarray:
        logits = predict_logits(model, x.astype("float32"))
        return softmax(calibrator.transform_logits(logits))

    for split_name, x_values, y_values in [
        ("train", splits["x_train"], splits["y_train"]),
        ("test", splits["x_test"], splits["y_test"]),
    ]:
        pfi = permutation_importance(
            predict_fn,
            x_values,
            y_values,
            splits["feature_names"],
            n_repeats=cfg["xai"]["permutation_repeats"],
            random_state=cfg["project"]["seed"],
        )
        pfi.to_csv(Path(cfg["paths"]["tables_dir"]) / f"pfi_{split_name}.csv", index=False)
        save_importance(
            pfi,
            Path(cfg["paths"]["figures_dir"]) / f"pfi_{split_name}.png",
            f"Permutation feature importance - {split_name}",
        )
        segments = segment_importance(pfi)
        segments.to_csv(Path(cfg["paths"]["tables_dir"]) / f"pfi_segments_{split_name}.csv", index=False)
        save_importance(
            segments,
            Path(cfg["paths"]["figures_dir"]) / f"pfi_segments_{split_name}.png",
            f"Segment-level PFI - {split_name}",
        )

    if not args.skip_lime:
        explainer = build_explainer(splits["x_train"], splits["feature_names"], label_names)
        lime_global = global_lime(
            explainer,
            predict_proba_fn,
            splits["x_test"],
            splits["feature_names"],
            cfg["xai"]["lime_global_instances"],
            cfg["xai"]["lime_num_features"],
            cfg["xai"]["lime_num_samples"],
        )
        lime_global.to_csv(Path(cfg["paths"]["tables_dir"]) / "lime_global.csv", index=False)
        save_lime_plot(lime_global, Path(cfg["paths"]["figures_dir"]) / "lime_global.png")
        save_local_lime(
            explainer,
            predict_proba_fn,
            splits["x_test"],
            cases,
            cfg["paths"]["lime_dir"],
            cfg["xai"]["lime_num_features"],
            cfg["xai"]["lime_num_samples"],
            cfg["xai"]["lime_local_cases_per_group"],
        )

    print("Pipeline finished. Metrics:", Path(cfg["paths"]["metrics_dir"]) / "test_metrics.csv")


if __name__ == "__main__":
    main()
