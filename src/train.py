from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import lightning as L
import matplotlib

matplotlib.use("Agg")

import optuna
import pandas as pd
import seaborn as sns
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from src.calibration import TemperatureScaler, softmax
from src.data import (
    clean_dataframe,
    feature_columns,
    load_raw_data,
    save_data_summary,
    save_processed_splits,
    split_and_scale,
)
from src.evaluation import CLASS_NAMES, predict_logits, save_evaluation, save_uncertainty_cases, uncertainty_table
from src.model import ARCHITECTURES, EEGDataModule, make_model, seed_everything
from src.xai import (
    build_lime_explainer,
    global_lime,
    permutation_importance,
    save_importance_plot,
    save_lime_plot,
    save_local_lime,
    segment_importance,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_PATH = DATA_DIR / "raw" / "data.csv"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"
MODELS_DIR = OUTPUTS_DIR / "models"
TABLES_DIR = OUTPUTS_DIR / "tables"
LIME_DIR = OUTPUTS_DIR / "lime"


def prepare_dirs() -> None:
    for path in [PROCESSED_DIR, FIGURES_DIR, METRICS_DIR, MODELS_DIR, TABLES_DIR, LIME_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def save_eda_figures(raw: pd.DataFrame, cleaned: pd.DataFrame, splits) -> None:
    features = feature_columns(cleaned)
    order = sorted(cleaned["y"].unique())

    plt.figure(figsize=(7, 4))
    sns.countplot(data=cleaned, x="y", order=order)
    plt.title("Class distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "class_distribution.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 6))
    for cls in order:
        sample = cleaned[cleaned["y"] == cls].iloc[0][features].to_numpy()
        plt.plot(sample, label=f"class {cls}", linewidth=1)
    plt.title("Example EEG signals by class")
    plt.xlabel("Time point")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "example_signals.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 6))
    for cls in order:
        mean_signal = cleaned[cleaned["y"] == cls][features].mean(axis=0).to_numpy()
        plt.plot(mean_signal, label=f"class {cls}", linewidth=1.5)
    plt.title("Mean EEG signal by class")
    plt.xlabel("Time point")
    plt.ylabel("Mean amplitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "mean_signals.png", dpi=180)
    plt.close()

    eda = raw[["y"]].copy()
    eda["signal_mean"] = raw[features].mean(axis=1)
    eda["signal_std"] = raw[features].std(axis=1)
    eda["signal_min"] = raw[features].min(axis=1)
    eda["signal_max"] = raw[features].max(axis=1)
    eda["signal_energy"] = (raw[features] ** 2).sum(axis=1)
    eda.describe().T.to_csv(TABLES_DIR / "signal_level_features_describe.csv")

    plt.figure(figsize=(7, 5))
    sns.boxplot(data=eda, x="y", y="signal_energy")
    plt.title("Signal energy by class")
    plt.xlabel("Class")
    plt.ylabel("Energy")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "signal_energy_by_class.png", dpi=180)
    plt.close()

    pca = PCA(n_components=2, random_state=42)
    x_pca = pca.fit_transform(splits.x_train)
    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(x_pca[:, 0], x_pca[:, 1], c=splits.y_train, s=8, alpha=0.55, cmap="tab10")
    plt.title("PCA of scaled training samples")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(scatter, label="Encoded class")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pca_train.png", dpi=180)
    plt.close()


def trainer_for(
    max_epochs: int,
    callbacks: list | None = None,
    checkpointing: bool = False,
    quiet: bool = False,
) -> L.Trainer:
    return L.Trainer(
        max_epochs=max_epochs,
        accelerator="cpu",
        devices=1,
        callbacks=callbacks or [],
        logger=False,
        enable_checkpointing=checkpointing,
        enable_progress_bar=not quiet,
        deterministic=True,
    )


def run_hpo(data_module: EEGDataModule, n_trials: int, max_epochs: int, seed: int, force: bool) -> dict:
    best_params_path = MODELS_DIR / "best_params.json"
    hyperparams_path = TABLES_DIR / "hyperparameter_search.csv"
    if best_params_path.exists() and hyperparams_path.exists() and not force:
        return json.loads(best_params_path.read_text(encoding="utf-8"))

    def objective(trial: optuna.Trial) -> float:
        arch_name = trial.suggest_categorical("architecture", list(ARCHITECTURES))
        overrides = {
            "hidden_dims": ARCHITECTURES[arch_name],
            "dropout": trial.suggest_float("dropout", 0.1, 0.5, step=0.1),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        }
        model = make_model(overrides)
        early = EarlyStopping(monitor="val_macro_f1", mode="max", patience=5)
        trainer = trainer_for(max_epochs=max_epochs, callbacks=[early], checkpointing=False, quiet=True)
        trainer.fit(model, datamodule=data_module)
        return float(trainer.callback_metrics["val_macro_f1"].detach().cpu().item())

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    study.trials_dataframe().to_csv(hyperparams_path, index=False)

    best_params = study.best_params
    best_params["hidden_dims"] = ARCHITECTURES[best_params.pop("architecture")]
    best_params_path.write_text(json.dumps(best_params, indent=2), encoding="utf-8")
    return best_params


def train_final_model(data_module: EEGDataModule, best_params: dict, max_epochs: int) -> torch.nn.Module:
    model = make_model(best_params)
    checkpoint = ModelCheckpoint(
        dirpath=MODELS_DIR,
        filename="eeg_mlp_best",
        monitor="val_macro_f1",
        mode="max",
        save_top_k=1,
    )
    early = EarlyStopping(monitor="val_macro_f1", mode="max", patience=10)
    trainer = trainer_for(max_epochs=max_epochs, callbacks=[early, checkpoint], checkpointing=True, quiet=True)
    trainer.fit(model, datamodule=data_module)
    if checkpoint.best_model_path:
        checkpoint_data = torch.load(checkpoint.best_model_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint_data["state_dict"])
    torch.save(model.state_dict(), MODELS_DIR / "eeg_mlp_state_dict.pt")
    return model


def run_xai(model, calibrator: TemperatureScaler, splits, run_lime: bool, seed: int) -> None:
    def predict_fn(x):
        logits = predict_logits(model, x.astype("float32"))
        return softmax(calibrator.transform_logits(logits)).argmax(axis=1)

    def predict_proba_fn(x):
        logits = predict_logits(model, x.astype("float32"))
        return softmax(calibrator.transform_logits(logits))

    for split_name, x_values, y_values in [
        ("train", splits.x_train, splits.y_train),
        ("test", splits.x_test, splits.y_test),
    ]:
        pfi = permutation_importance(predict_fn, x_values, y_values, splits.feature_names, seed=seed, n_repeats=3)
        pfi.to_csv(TABLES_DIR / f"pfi_{split_name}.csv", index=False)
        save_importance_plot(
            pfi,
            FIGURES_DIR / f"pfi_{split_name}.png",
            f"Permutation feature importance - {split_name}",
        )

        segments = segment_importance(pfi)
        segments.to_csv(TABLES_DIR / f"pfi_segments_{split_name}.csv", index=False)
        save_importance_plot(
            segments,
            FIGURES_DIR / f"pfi_segments_{split_name}.png",
            f"Segment-level PFI - {split_name}",
        )

    if run_lime:
        cases = {
            name: pd.read_csv(TABLES_DIR / f"{name}.csv")
            for name in ["most_confident_correct", "most_uncertain_correct", "most_uncertain_wrong"]
        }
        explainer = build_lime_explainer(splits.x_train, splits.feature_names, seed=seed)
        lime_global = global_lime(
            explainer,
            predict_proba_fn,
            splits.x_test,
            splits.feature_names,
            seed=seed,
            n_instances=120,
        )
        lime_global.to_csv(TABLES_DIR / "lime_global.csv", index=False)
        save_lime_plot(lime_global, FIGURES_DIR)
        save_local_lime(explainer, predict_proba_fn, splits.x_test, cases, LIME_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the reproducible EEG multiclass classification pipeline.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--max-epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--force-hpo", action="store_true")
    parser.add_argument("--skip-lime", action="store_true")
    args = parser.parse_args()

    prepare_dirs()
    seed_everything(args.seed)
    raw = load_raw_data(RAW_PATH)
    cleaned = clean_dataframe(raw)
    splits = split_and_scale(cleaned, random_state=args.seed)
    save_processed_splits(splits, PROCESSED_DIR)
    save_data_summary(cleaned, splits.feature_names, TABLES_DIR)
    save_eda_figures(raw, cleaned, splits)

    data_module = EEGDataModule(splits, batch_size=args.batch_size)
    best_params = run_hpo(
        data_module,
        n_trials=args.n_trials,
        max_epochs=args.max_epochs,
        seed=args.seed,
        force=args.force_hpo,
    )
    model = train_final_model(data_module, best_params, max_epochs=args.max_epochs)

    val_logits = predict_logits(model, splits.x_val)
    test_logits = predict_logits(model, splits.x_test)
    calibrator = TemperatureScaler().fit(val_logits, splits.y_val)
    joblib.dump(calibrator, MODELS_DIR / "temperature_scaler.joblib")

    probs_before = softmax(test_logits)
    probs_after = softmax(calibrator.transform_logits(test_logits))
    save_evaluation(splits.y_test, probs_before, probs_after, METRICS_DIR, FIGURES_DIR)

    predictions = pd.DataFrame(probs_after, columns=[f"prob_{name}" for name in CLASS_NAMES])
    predictions.insert(0, "pred", probs_after.argmax(axis=1))
    predictions.insert(0, "true", splits.y_test)
    predictions.to_csv(TABLES_DIR / "test_predictions_calibrated.csv", index=False)

    cases = save_uncertainty_cases(uncertainty_table(splits.y_test, probs_after), TABLES_DIR)
    run_xai(model, calibrator, splits, run_lime=not args.skip_lime, seed=args.seed)

    run_config = {
        "seed": args.seed,
        "n_trials": args.n_trials,
        "max_epochs": args.max_epochs,
        "batch_size": args.batch_size,
        "best_params": best_params,
        "temperature": calibrator.temperature,
        "uncertainty_case_counts": {name: len(group) for name, group in cases.items()},
    }
    (MODELS_DIR / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    print(json.dumps(run_config, indent=2))


if __name__ == "__main__":
    main()
