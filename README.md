# GradeScope AI · Student Academic Performance Prediction

**A complete deep learning coursework project with an interactive Streamlit application, DNN training, a Random Forest comparison, feature selection, exploratory data analysis, and evaluation reports.**

## What is new in v2

- **Rebuilt dark dashboard** — professional branded UI, five navigation sections, Plotly charts, responsive layout.
- **Predictor** — a quick profile and optional advanced inputs covering every training feature, scenario examples, side-by-side probability display, and downloadable prediction JSON.
- **Model Lab** — real held-out test metrics, per-model confusion matrices, learning curves, hyperparameter trials, Random Forest feature importance.
- **Explorer** — interactive grade distributions, correlation matrix, study-time charts, and a filtered CSV viewer.
- **Robust UCI downloads** — handles `student-por.csv` / `student-mat.csv` in the top ZIP, subfolders, and inside the nested `student.zip`. Saves validated CSVs locally.
- **Earlier-year prediction by default** — excludes `G1` and `G2` unless explicitly enabled to avoid performance inflation from grades only available later.
- **Windows installation helpers** — creates `.venv` and temporary installer files on **your project drive** (helpful when C: is nearly full).
- **Reproducible training** — seed 42, train-only preprocessing and feature selection, validation-only architecture selection, held-out test evaluation.

## Quick start (Windows)

**Prerequisite:** Python **3.11 or 3.12**, internet connection for the initial UCI dataset download, and preferably 3–5 GB of free disk space on the drive containing this project. TensorFlow is large.

### Easiest method

1. Extract the project ZIP to a folder on a drive with sufficient storage, for example `D:\DNNN\GradeScope_AI`.
2. Double-click `setup_windows.bat` **once**. This creates `.venv` locally and installs dependencies without retaining pip's large download cache.
3. Double-click `train_models.bat` **once** to train the Portuguese (649-row) models. Models and actual metrics are stored under `artifacts/por/early_prediction/`.
4. Double-click `run_dashboard.bat` to open the Streamlit UI at `http://localhost:8501`.

### Run from VS Code / PowerShell

```powershell
cd D:\DNNN\GradeScope_AI
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --no-cache-dir -r requirements.txt
.\.venv\Scripts\python.exe train.py --subject por --tune
.\.venv\Scripts\python.exe -m streamlit run app.py
```

You can also launch the app **before training**, visit **Predict** or **Model Lab**, and click **Train model inside the app**. The training button runs locally and displays the console output. Training must finish successfully before predictions are available.

> If you are working on drive D: with little C: space, use `setup_windows.bat`. It puts both the Python environment and pip's temporary installation directory on D:. Do not run `pip install` without the local `.venv` if you want to keep installations out of the global Python directory.

## Exact assignment coverage

| Requirement | Implementation |
|---|---|
| Real dataset | UCI Student Performance (`por`, 649; `mat`, 395) |
| Data preprocessing | Median/mode imputation, scaling, one-hot encoding |
| Exploratory data analysis | `eda.py`, Explorer tab, charts, CSV tables |
| Feature selection | Training-only ANOVA `SelectKBest` on encoded features |
| Deep neural network | 2 hidden Dense layers, ReLU, batch normalization, dropout, softmax |
| Hyperparameter tuning | Validation macro-F1 over 3 DNN architecture configs using `--tune` |
| Traditional ML baseline | Random Forest on same feature representation and same held-out test split |
| Performance evaluation | Accuracy, balanced accuracy, precision, recall, macro/weighted F1, confusion matrices, learning history |
| Streamlit deployment | `app.py` with interactive inputs, predicted grade category and model scores |
| Model artifacts | Saved `.keras`, `.joblib`, `.json`, `.csv` and evaluation `.png` files |

## Grade mapping

| Grade `G3` | Output class |
|---|---|
| `0–9` | Low |
| `10–14` | Medium |
| `15–20` | High |

These thresholds are **project-defined**, not a validated student placement or employability metric.

## Training options

```powershell
# Recommended early estimate (no first/second-period grades)
python train.py --subject por --tune

# Same experiment on Mathematics records
python train.py --subject mat --tune

# Easier late-year prediction — includes G1 and G2
python train.py --subject por --tune --include-prior-grades

# Faster single-configuration baseline
python train.py --subject por --epochs 50

# Generate descriptive EDA CSV tables
python eda.py --subject por
```

**Important:** Use `\.venv\Scripts\python.exe` in place of `python` if your global interpreter differs from the one you installed into your virtual environment.

Each subject/stage combination saves independently:

```
artifacts/
  por/
    early_prediction/
      dnn.keras
      pipeline.joblib
      metadata.json
      metrics.json
      feature_importance.csv
      plots/
        grade_distribution.png
        training_history.png
        confusion_matrices.png
        feature_importance.png
    with_grades/          # only if explicitly trained
  mat/
    early_prediction/     # only if explicitly trained
```

## Downloading data manually (offline fallback)

If the automatic download fails:

1. Open [UCI Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance).
2. Download the 39.8 KB outer ZIP. **It may contain another file named `student.zip`**.
3. Open that inner ZIP and extract `student-por.csv` and/or `student-mat.csv`.
4. Place files directly inside the project's `data` directory (e.g. `D:\DNNN\GradeScope_AI\data\student-por.csv`).
5. Re-run training.

The data loader accepts a valid local CSV without network access.

## How this experiment avoids leakage

1. Stratified development/test split (80/20), with separate validation inside development.
2. Fit imputation, scaling, one-hot categories and **supervised feature selection only on the training subset** during tuning.
3. Choose the DNN architecture by *validation* macro F1, never test F1.
4. Refit feature transforms using the **development split**, train final models, evaluate each on the **previously untouched test split** exactly once.
5. Default model excludes `G1` / `G2` because these are earlier reporting-period grades strongly correlated with final grade `G3`; including them is a different, later-stage task.

The split is not an external validation. All metrics from a small historical dataset should be treated as approximate, and repeating stratified CV would improve confidence.

## Reproducibility and honest evaluation

- **No trained models or fabricated metrics are included in the ZIP.** Actual outputs are generated after running `train.py` against the real UCI data on your computer.
- The first dataset download requires a network connection. All subsequent runs can use the saved local CSV.
- The DNN and Random Forest may have substantially different results: a more complex model is **not guaranteed** to outperform the baseline.
- The predictor's top softmax score is **uncalibrated**; it is not a guaranteed probability of an individual outcome.
- Random Forest feature importance describes a particular model; it does **not** establish feature causality or DNN interpretability.

## Project structure

```
GradeScope_AI/
  app.py                  Streamlit UI (five sections)
  assets/theme.css        Custom dashboard styling
  .streamlit/config.toml  Streamlit dark theme
  train.py                Full training, tuning, evaluation, artifact export
  data_utils.py           Official dataset loading, validation and feature definitions
  predict.py              Saved pipeline inference
  eda.py                  Tabular EDA exports
  requirements.txt        Python dependencies
  setup_windows.bat       Installs packages with local drive temp and venv
  train_models.bat        Trains recommended Portuguese models
  run_dashboard.bat       Launches web UI
  PROJECT_REPORT.md       Submission report outline and experiment description
  tests/                  Automated dataset and ML pipeline checks
  data/README.md          UCI dataset download help
  artifacts/              Created after training, not included
  reports/                Created by EDA, not included
```

## Common errors

**`No space left on device`** — The full TensorFlow install needs substantial space. Extract on D:, run `setup_windows.bat`, or use a virtual environment and temp directory on D:. Keep `--no-cache-dir`.

**`student-por.csv is not in archive`** — Fixed in `data_utils.py`. It recurses into the UCI `student.zip` if present. Make sure you are working from the newly extracted **GradeScope_AI** folder, not the older project folder.

**`expected 'except' or 'finally' block`** — That was caused by an incorrect manual edit to an older project. The new files have been syntax-checked; extract them instead of mixing old and new files.

**`ModuleNotFoundError: streamlit / tensorflow`** — Use `\.venv\Scripts\python.exe -m ...` consistently after installing into the local environment.

**App says `training required`** — It means `artifacts/<subject>/<mode>/` does not contain trained models. Train the same subject and stage selected in the sidebar.

**Very low scores** — This is possible with a small multi-class dataset, especially when earlier grades are excluded. Review class imbalance, the confusion matrices, and the selected model configuration, but do **not** optimize against the held-out test set.

## Responsible use

The dataset was gathered from two Portuguese **secondary schools**, and is unsuitable for evaluating present-day students' employability. Some features contain potentially sensitive demographic or household information. This program is an **educational prototype only**. It must not be used to make consequential educational, hiring, financial, disciplinary, or admission decisions about anyone.

## Data attribution

Cortez, P. (2008). *Student Performance*. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T.

UCI hosts the dataset under **Creative Commons Attribution 4.0 (CC BY 4.0)**. This code automatically obtains the data from the official repository rather than redistributing it in this ZIP.
