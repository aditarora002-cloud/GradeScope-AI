"""Load saved preprocessing, DNN, and Random Forest for inference."""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from data_utils import CLASSES, ROOT


def artifact_dir(subject: str = "por", with_grades: bool = False) -> Path:
    if subject not in ("por", "mat"):
        raise ValueError("Only 'por' and 'mat' subjects are supported")
    return ROOT / "artifacts" / subject / ("with_grades" if with_grades else "early_prediction")


def artifacts_ready(subject: str = "por", with_grades: bool = False) -> bool:
    directory = artifact_dir(subject, with_grades)
    return all((directory / name).is_file() for name in ("pipeline.joblib", "dnn.keras", "metadata.json", "metrics.json"))


class Predictor:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.metadata = json.loads((self.directory / "metadata.json").read_text(encoding="utf-8"))
        self.metrics = json.loads((self.directory / "metrics.json").read_text(encoding="utf-8"))
        self.bundle = joblib.load(self.directory / "pipeline.joblib")
        if self.bundle["features"] != self.metadata["feature_columns"]:
            raise ValueError("Metadata and trained pipeline input columns do not match")
        self._dnn = None

    @property
    def dnn(self):
        if self._dnn is None:
            import tensorflow as tf
            self._dnn = tf.keras.models.load_model(self.directory / "dnn.keras", compile=False)
        return self._dnn

    def predict(self, profile: dict, model_name: str = "DNN") -> dict:
        fields = self.bundle["features"]
        missing = [field for field in fields if field not in profile]
        if missing:
            raise ValueError(f"Missing required input fields: {missing}")
        input_df = pd.DataFrame([{name: profile[name] for name in fields}], columns=fields)
        encoded = self.bundle["preprocessor"].transform(input_df)
        x = self.bundle["selector"].transform(encoded)
        if model_name == "DNN":
            probabilities = self.dnn.predict(x, verbose=0)[0]
        elif model_name == "Random Forest":
            forest = self.bundle["random_forest"]
            raw = forest.predict_proba(x)[0]
            probabilities = np.zeros(len(CLASSES))
            probabilities[forest.classes_] = raw
        else:
            raise ValueError("model_name must be 'DNN' or 'Random Forest'")
        index = int(np.argmax(probabilities))
        return {
            "category": CLASSES[index],
            "confidence": float(probabilities[index]),
            "probabilities": {name: float(probabilities[i]) for i, name in enumerate(CLASSES)},
            "model": model_name,
            "input": {name: profile[name] for name in fields},
        }
