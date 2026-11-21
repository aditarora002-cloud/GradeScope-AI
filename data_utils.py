"""Official UCI dataset loader, validation, and grade labeling.

No artificial rows are used for real training or evaluation.
"""
from __future__ import annotations

import io
from pathlib import Path
import urllib.request
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
DATASET_URL = "https://archive.ics.uci.edu/dataset/320/student+performance"
CLASSES = ["Low", "Medium", "High"]
TARGET = "G3"
RAW_FEATURES = [
    "school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu",
    "Mjob", "Fjob", "reason", "guardian", "traveltime", "studytime",
    "failures", "schoolsup", "famsup", "paid", "activities", "nursery",
    "higher", "internet", "romantic", "famrel", "freetime", "goout",
    "Dalc", "Walc", "health", "absences", "G1", "G2",
]
NUMERIC_FEATURES = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures", "famrel",
    "freetime", "goout", "Dalc", "Walc", "health", "absences", "G1", "G2",
]
FEATURE_GROUPS = {
    "Academics": ["studytime", "failures", "absences", "schoolsup", "famsup", "paid", "higher", "G1", "G2"],
    "Daily life": ["activities", "internet", "romantic", "freetime", "goout", "health", "Dalc", "Walc", "traveltime"],
    "Family & background": ["school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu", "Mjob", "Fjob", "reason", "guardian", "nursery", "famrel"],
}
FEATURE_LABELS = {
    "school": "School", "sex": "Sex (dataset field)", "age": "Age", "address": "Home location",
    "famsize": "Family size", "Pstatus": "Parents living together", "Medu": "Mother's education",
    "Fedu": "Father's education", "Mjob": "Mother's occupation", "Fjob": "Father's occupation",
    "reason": "School choice reason", "guardian": "Primary guardian", "traveltime": "Journey to school",
    "studytime": "Weekly study time", "failures": "Past class failures", "schoolsup": "Extra school support",
    "famsup": "Family education support", "paid": "Extra paid classes", "activities": "Extracurricular activities",
    "nursery": "Attended nursery", "higher": "Wants higher education", "internet": "Home internet access",
    "romantic": "In a relationship", "famrel": "Family relationships (1–5)", "freetime": "Free time (1–5)",
    "goout": "Going out with friends (1–5)", "Dalc": "Weekday alcohol use (1–5)",
    "Walc": "Weekend alcohol use (1–5)", "health": "Self-reported health (1–5)",
    "absences": "School absences", "G1": "First-period grade", "G2": "Second-period grade",
}
# Options encode the *original UCI values*; friendly labels are only for the UI.
CATEGORY_OPTIONS = {
    "school": {"GP": "Gabriel Pereira", "MS": "Mousinho da Silveira"},
    "sex": {"F": "Female", "M": "Male"},
    "address": {"U": "Urban", "R": "Rural"},
    "famsize": {"LE3": "3 or fewer", "GT3": "More than 3"},
    "Pstatus": {"T": "Together", "A": "Apart"},
    "Mjob": {"teacher": "Teacher", "health": "Healthcare", "services": "Services", "at_home": "At home", "other": "Other"},
    "Fjob": {"teacher": "Teacher", "health": "Healthcare", "services": "Services", "at_home": "At home", "other": "Other"},
    "reason": {"home": "Close to home", "reputation": "Reputation", "course": "Course", "other": "Other"},
    "guardian": {"mother": "Mother", "father": "Father", "other": "Other"},
    "schoolsup": {"yes": "Yes", "no": "No"}, "famsup": {"yes": "Yes", "no": "No"},
    "paid": {"yes": "Yes", "no": "No"}, "activities": {"yes": "Yes", "no": "No"},
    "nursery": {"yes": "Yes", "no": "No"}, "higher": {"yes": "Yes", "no": "No"},
    "internet": {"yes": "Yes", "no": "No"}, "romantic": {"yes": "Yes", "no": "No"},
}
NUMERIC_BOUNDS = {
    "age": (15, 22), "Medu": (0, 4), "Fedu": (0, 4), "traveltime": (1, 4),
    "studytime": (1, 4), "failures": (0, 4), "famrel": (1, 5), "freetime": (1, 5),
    "goout": (1, 5), "Dalc": (1, 5), "Walc": (1, 5), "health": (1, 5),
    "absences": (0, 93), "G1": (0, 20), "G2": (0, 20),
}
STUDY_LABELS = {1: "<2 hours", 2: "2–5 hours", 3: "5–10 hours", 4: ">10 hours"}
TRAVEL_LABELS = {1: "<15 min", 2: "15–30 min", 3: "30–60 min", 4: ">60 min"}
EDUCATION_LABELS = {0: "None", 1: "Primary", 2: "Middle school", 3: "Secondary", 4: "Higher education"}


def extract_csv_from_archive(blob: bytes, subject: str, max_depth: int = 3) -> bytes:
    """Locate the requested CSV even if it is nested inside student.zip."""
    wanted = f"student-{subject}.csv"

    def walk(data: bytes, depth: int) -> bytes | None:
        if depth > max_depth:
            return None
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            files = [i for i in zf.infolist() if not i.is_dir()]
            for item in files:
                if Path(item.filename).name.lower() == wanted and item.file_size < 5_000_000:
                    return zf.read(item)
            for item in files:
                if item.filename.lower().endswith(".zip") and item.file_size < 10_000_000:
                    found = walk(zf.read(item), depth + 1)
                    if found is not None:
                        return found
        return None

    result = walk(blob, 0)
    if result is None:
        raise FileNotFoundError(f"{wanted} wasn't found in the UCI archive or its nested ZIPs")
    return result


def load_data(subject: str = "por", data_dir: Path | None = None) -> pd.DataFrame:
    """Read local CSV or download once from UCI; never silently generate fake data."""
    if subject not in ("por", "mat"):
        raise ValueError("subject must be 'por' (Portuguese) or 'mat' (Mathematics)")
    folder = Path(data_dir) if data_dir is not None else DATA_DIR
    folder.mkdir(parents=True, exist_ok=True)
    csv_path = folder / f"student-{subject}.csv"
    if not csv_path.exists():
        try:
            request = urllib.request.Request(DATA_URL, headers={"User-Agent": "GradeScopeAI/2.0 (educational use)"})
            with urllib.request.urlopen(request, timeout=40) as response:
                archive = response.read()
            csv_bytes = extract_csv_from_archive(archive, subject)
            # Validate data in memory before storing anything.
            validate_data(pd.read_csv(io.BytesIO(csv_bytes), sep=";"))
            csv_path.write_bytes(csv_bytes)
        except Exception as exc:
            raise RuntimeError(
                "Could not obtain the UCI dataset automatically. Download the dataset from "
                f"{DATASET_URL}, unzip student.zip if necessary, and put student-{subject}.csv "
                f"inside '{folder.resolve()}'. Original error: {exc}"
            ) from exc
    df = pd.read_csv(csv_path, sep=";")
    validate_data(df)
    return df


def validate_data(df: pd.DataFrame) -> None:
    missing = sorted(set(RAW_FEATURES + [TARGET]) - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing UCI columns: {missing}. Expected a semicolon-separated student-{chr(112)}or.csv or student-mat.csv")
    if not pd.to_numeric(df[TARGET], errors="coerce").between(0, 20).all():
        raise ValueError("G3 must contain numeric grades between 0 and 20")


def grade_category(grades: pd.Series | np.ndarray | list | int) -> pd.Series | str:
    """Low: 0–9, Medium: 10–14, High: 15–20 (project-defined bands)."""
    if np.isscalar(grades):
        value = int(grades)
        if not 0 <= value <= 20:
            raise ValueError("Grade must be between 0 and 20")
        return CLASSES[0 if value <= 9 else (1 if value <= 14 else 2)]
    s = pd.Series(grades)
    return pd.cut(s, bins=[-0.5, 9.5, 14.5, 20.5], labels=CLASSES).astype(str)


def training_features(df: pd.DataFrame, include_prior_grades: bool = False) -> pd.DataFrame:
    fields = [f for f in RAW_FEATURES if include_prior_grades or f not in ("G1", "G2")]
    return df.loc[:, fields].copy()


def input_defaults(df: pd.DataFrame, features: list[str]) -> dict:
    defaults = {}
    for field in features:
        series = df[field].dropna()
        if series.empty:
            defaults[field] = 0 if field in NUMERIC_FEATURES else "other"
        elif field in NUMERIC_FEATURES:
            defaults[field] = int(round(float(series.median())))
        else:
            defaults[field] = str(series.mode().iloc[0])
    return defaults


def profile_from_example(defaults: dict, label: str) -> dict:
    """Illustrative input scenarios; not actual UCI records or ground truth."""
    profile = dict(defaults)
    if label == "Consistent learner":
        profile.update(studytime=4, failures=0, absences=1, higher="yes", schoolsup="yes", G1=16, G2=17)
    elif label == "Needs support":
        profile.update(studytime=1, failures=2, absences=18, higher="no", schoolsup="no", G1=7, G2=8)
    return profile
