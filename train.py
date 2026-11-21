"""Train and evaluate GradeScope AI on an untouched holdout test split.

Usage:
    python train.py --subject por --tune
    python train.py --subject mat --epochs 60
    python train.py --subject por --include-prior-grades --tune
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_utils import (CLASSES, DATASET_URL, NUMERIC_FEATURES, grade_category, input_defaults,
                        load_data, training_features)

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "artifacts"
SEED = 42


def make_preprocessor(features: list[str]) -> ColumnTransformer:
    numeric = [col for col in features if col in NUMERIC_FEATURES]
    categorical = [col for col in features if col not in NUMERIC_FEATURES]
    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), numeric),
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), categorical),
        ],
        remainder="drop",
        sparse_threshold=0,
    )


def make_selector(x: np.ndarray, y: np.ndarray, k: int):
    # Constant one-hot columns can arise in small samples; sklearn handles them by
    # assigning an uninformative score. Feature selection sees training data only.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.feature_selection")
        selector = SelectKBest(score_func=f_classif, k=min(k, x.shape[1]))
        return selector, selector.fit_transform(x, y)


def build_dnn(n_features: int, width: int, dropout: float, learning_rate: float):
    import tensorflow as tf

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_features,)),
        tf.keras.layers.Dense(width, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(max(width // 2, 16), activation="relu"),
        tf.keras.layers.Dropout(dropout * 0.8),
        tf.keras.layers.Dense(len(CLASSES), activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def callbacks():
    import tensorflow as tf
    return [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5, factor=0.5, min_lr=1e-5),
    ]


def weighted_classes(y: np.ndarray) -> dict:
    counts = np.bincount(y, minlength=len(CLASSES))
    total = len(y)
    return {i: float(total / (len(CLASSES) * max(int(counts[i]), 1))) for i in range(len(CLASSES))}


def report_metrics(y_true, y_pred, probs=None) -> dict:
    payload = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=range(len(CLASSES))).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=range(len(CLASSES)), target_names=CLASSES,
            output_dict=True, zero_division=0,
        ),
    }
    if probs is not None:
        payload["mean_max_class_probability"] = float(np.max(probs, axis=1).mean())
    return payload


def save_diagnostics(out: Path, df: pd.DataFrame, history: dict, evaluation: dict, importance: pd.DataFrame) -> None:
    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    colors = ["#F59E0B", "#38BDF8", "#A78BFA"]

    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    counts = grade_category(df["G3"]).value_counts().reindex(CLASSES, fill_value=0)
    ax.bar(CLASSES, counts.values, color=colors, width=.56)
    for i, v in enumerate(counts.values): ax.text(i, v + 2, str(v), ha="center", fontsize=11)
    ax.set_title("Final grade categories in the UCI sample")
    ax.set_ylabel("Number of students")
    ax.set_ylim(0, max(counts.max() * 1.18, 1))
    fig.tight_layout(); fig.savefig(plots / "grade_distribution.png", dpi=170); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for key, axis, ttl in [("loss", axes[0], "Loss"), ("accuracy", axes[1], "Accuracy")]:
        axis.plot(history.get(key, []), color="#7C3AED", label="Train", linewidth=2.5)
        axis.plot(history.get("val_" + key, []), color="#0891B2", label="Validation", linewidth=2.5)
        axis.set_title(ttl); axis.set_xlabel("Epoch"); axis.legend()
    fig.tight_layout(); fig.savefig(plots / "training_history.png", dpi=170); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, model_name in zip(axes, ("DNN", "Random Forest")):
        cm = np.asarray(evaluation[model_name]["confusion_matrix"])
        ax.imshow(cm, cmap="Blues")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="#111827")
        ax.set_xticks(range(3), CLASSES); ax.set_yticks(range(3), CLASSES)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(model_name)
    fig.tight_layout(); fig.savefig(plots / "confusion_matrices.png", dpi=170); plt.close(fig)

    if not importance.empty:
        top = importance.head(15).iloc[::-1]
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(top["feature"], top["importance"], color="#7C3AED")
        ax.set_title("Random Forest feature importance (selected features)")
        fig.tight_layout(); fig.savefig(plots / "feature_importance.png", dpi=170); plt.close(fig)


def train_project(subject="por", epochs=70, tune=False, include_prior_grades=False,
                  feature_count=40, batch_size=32, data_dir=None, artifacts_root=DEFAULT_OUTPUT):
    import tensorflow as tf

    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
    print("[1/7] Loading official UCI records...", flush=True)
    df = load_data(subject, data_dir=data_dir)
    X = training_features(df, include_prior_grades=include_prior_grades)
    y = grade_category(df["G3"]).map({name: i for i, name in enumerate(CLASSES)}).to_numpy(dtype=int)
    features = X.columns.tolist()
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_dev, y_dev, test_size=0.20, random_state=SEED, stratify=y_dev)
    print(f"        {len(df)} records | {len(features)} raw features | "
          f"{len(X_fit)} fit / {len(X_val)} validation / {len(X_test)} test", flush=True)

    print("[2/7] Preprocessing and selecting training-only features...", flush=True)
    prep_tune = make_preprocessor(features)
    fit_encoded = prep_tune.fit_transform(X_fit)
    val_encoded = prep_tune.transform(X_val)
    selector_tune, fit_array = make_selector(fit_encoded, y_fit, feature_count)
    val_array = selector_tune.transform(val_encoded)

    configs = [{"width": 64, "dropout": 0.25, "learning_rate": 0.001}]
    if tune:
        configs = [
            {"width": 32, "dropout": 0.20, "learning_rate": 0.001},
            {"width": 64, "dropout": 0.25, "learning_rate": 0.001},
            {"width": 128, "dropout": 0.35, "learning_rate": 0.0005},
        ]
    best_score = -1.0
    best_config = configs[0]
    tuning_results = []
    print(f"[3/7] Selecting DNN architecture ({len(configs)} configuration(s))...", flush=True)
    for index, config in enumerate(configs, 1):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(SEED + index)
        model = build_dnn(fit_array.shape[1], **config)
        history = model.fit(
            fit_array, y_fit, validation_data=(val_array, y_val),
            epochs=epochs, batch_size=batch_size, callbacks=callbacks(), verbose=0,
            class_weight=weighted_classes(y_fit),
        )
        probs = model.predict(val_array, verbose=0)
        score = float(f1_score(y_val, probs.argmax(1), average="macro", zero_division=0))
        result = {**config, "validation_macro_f1": score, "epochs_completed": len(history.history["loss"])}
        tuning_results.append(result)
        print(f"        Trial {index}/{len(configs)}: macro F1={score:.3f} | "
              f"{result['epochs_completed']} epochs | {config}", flush=True)
        if score > best_score:
            best_score, best_config = score, dict(config)
        del model

    print("[4/7] Re-fitting transforms on development split (test never seen)...", flush=True)
    preprocessor = make_preprocessor(features)
    dev_encoded = preprocessor.fit_transform(X_dev)
    test_encoded = preprocessor.transform(X_test)
    selector, dev_array = make_selector(dev_encoded, y_dev, feature_count)
    test_array = selector.transform(test_encoded)
    all_names = np.asarray(preprocessor.get_feature_names_out())
    selected_names = all_names[selector.get_support()].tolist()

    print("[5/7] Training final DNN and Random Forest...", flush=True)
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)
    dnn = build_dnn(dev_array.shape[1], **best_config)
    # This validation slice is strictly drawn from development data; no test usage.
    x_final_fit, x_final_val, y_final_fit, y_final_val = train_test_split(
        dev_array, y_dev, test_size=0.16, random_state=SEED + 1, stratify=y_dev)
    final_history = dnn.fit(
        x_final_fit, y_final_fit,
        validation_data=(x_final_val, y_final_val),
        epochs=epochs, batch_size=batch_size, callbacks=callbacks(), verbose=0,
        class_weight=weighted_classes(y_final_fit),
    )
    forest = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=2, class_weight="balanced",
        max_features="sqrt", random_state=SEED, n_jobs=-1,
    )
    forest.fit(dev_array, y_dev)

    print("[6/7] Evaluating BOTH models on the held-out test split...", flush=True)
    dnn_probs = dnn.predict(test_array, verbose=0)
    rf_probs = forest.predict_proba(test_array)
    # All three label classes appear in the official dataset; preserve ordering nevertheless.
    if list(forest.classes_) != list(range(len(CLASSES))):
        rf_aligned = np.zeros((len(test_array), len(CLASSES)), dtype=float)
        rf_aligned[:, forest.classes_] = rf_probs
        rf_probs = rf_aligned
    evaluation = {
        "DNN": report_metrics(y_test, dnn_probs.argmax(1), dnn_probs),
        "Random Forest": report_metrics(y_test, rf_probs.argmax(1), rf_probs),
    }
    for name, report in evaluation.items():
        print(f"        {name}: accuracy={report['accuracy']:.3f}; "
              f"macro F1={report['macro_f1']:.3f}; balanced accuracy={report['balanced_accuracy']:.3f}", flush=True)

    print("[7/7] Saving usable models, metrics and plots...", flush=True)
    out = Path(artifacts_root) / subject / ("with_grades" if include_prior_grades else "early_prediction")
    out.mkdir(parents=True, exist_ok=True)
    dnn.save(out / "dnn.keras")
    joblib.dump({
        "preprocessor": preprocessor,
        "selector": selector,
        "random_forest": forest,
        "features": features,
        "classes": CLASSES,
    }, out / "pipeline.joblib")
    feature_imp = pd.DataFrame({
        "feature": selected_names,
        "importance": forest.feature_importances_,
    }).sort_values("importance", ascending=False, ignore_index=True)
    feature_imp.to_csv(out / "feature_importance.csv", index=False)
    metadata = {
        "project": "GradeScope AI", "subject": subject, "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": DATASET_URL, "total_records": int(len(df)),
        "train_count": int(len(X_fit)), "validation_count": int(len(X_val)), "test_count": int(len(X_test)),
        "train_development_count": int(len(X_dev)), "feature_columns": features,
        "encoded_features": int(dev_encoded.shape[1]), "selected_features": selected_names,
        "input_defaults": input_defaults(df, features), "with_prior_grades": include_prior_grades,
        "labels": CLASSES, "grade_bands": {"Low": "0–9", "Medium": "10–14", "High": "15–20"},
        "best_dnn_config": best_config,
        "disclaimer": "Educational estimates on a small historical school dataset; not employability or hiring decisions.",
    }
    metrics = {
        "models": evaluation,
        "tuning_results": tuning_results,
        "training_history": {k: [float(v) for v in values] for k, values in final_history.history.items()},
        "class_distribution": {label: int((y == i).sum()) for i, label in enumerate(CLASSES)},
        "test_labels": [CLASSES[int(i)] for i in y_test],
        "test_predictions_dnn": [CLASSES[int(i)] for i in dnn_probs.argmax(1)],
        "test_predictions_rf": [CLASSES[int(i)] for i in rf_probs.argmax(1)],
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    save_diagnostics(out, df, metrics["training_history"], evaluation, feature_imp)
    print(f"\nDone. Artifacts saved in: {out.resolve()}", flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description="Train GradeScope AI (DNN vs Random Forest)")
    parser.add_argument("--subject", choices=["por", "mat"], default="por", help="por=649 records, mat=395 records")
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--feature-count", type=int, default=40)
    parser.add_argument("--tune", action="store_true", help="Compare 3 DNN configurations on validation data")
    parser.add_argument("--include-prior-grades", "--include-g1-g2", dest="include_prior_grades",
                        action="store_true", help="Include G1 and G2 (late-year prediction only)")
    args = parser.parse_args()
    if not 5 <= args.epochs <= 500 or not 5 <= args.feature_count <= 500:
        parser.error("epochs must be 5–500 and feature-count must be 5–500")
    train_project(subject=args.subject, epochs=args.epochs, batch_size=args.batch_size,
                  tune=args.tune, include_prior_grades=args.include_prior_grades,
                  feature_count=args.feature_count)


if __name__ == "__main__":
    main()
