"""Export descriptive EDA tables independently from training.

Usage: python eda.py --subject por
"""
from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd
from data_utils import grade_category, load_data, ROOT


def build_report(subject="por", out_dir=None):
    df = load_data(subject)
    out = Path(out_dir) if out_dir else ROOT / "reports" / subject
    out.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing_count": df.isna().sum(),
        "unique_values": df.nunique(),
    })
    summary.to_csv(out / "column_quality.csv")
    df.describe(include="all").T.to_csv(out / "descriptive_statistics.csv")
    grade_counts = grade_category(df["G3"]).value_counts().reindex(["Low", "Medium", "High"], fill_value=0)
    grade_counts.rename_axis("Category").rename("Students").to_csv(out / "grade_categories.csv")
    numeric = df.select_dtypes(include="number")
    numeric.corr(numeric_only=True).to_csv(out / "correlation_matrix.csv")
    print(f"EDA reports saved in {out.resolve()}")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--subject", choices=["por", "mat"], default="por")
    args = p.parse_args()
    build_report(args.subject)
