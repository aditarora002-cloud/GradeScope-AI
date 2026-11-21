# Project Report — GradeScope AI
## Student Academic Performance Prediction Using Deep Neural Networks

> Use this as the written submission template. **Populate the results section from the generated `metrics.json`**. No scores are fabricated or prefilled.

### 1. Abstract

GradeScope AI studies whether student academic performance categories can be estimated from educational, family, and behavioral characteristics. The project applies a Deep Neural Network (DNN) to the UCI Student Performance dataset and compares it to a Random Forest classifier using a separate, held-out test set. A five-section Streamlit dashboard supports profile-based inference, exploratory visualizations, evaluation, and a reproducible methodology.

### 2. Problem statement

Predict students' final grade category from a historical secondary-school dataset with two optional modes: (a) early-year prediction **without** previous-period grades G1 and G2 (recommended default), and (b) later-year prediction using G1 and G2. Three categories are defined for coursework: Low (0–9), Medium (10–14), High (15–20).

**Boundary:** These labels describe grades, not employability. The model is not suitable for deciding employment suitability or other consequential outcomes.

### 3. Dataset

- **Source:** UCI Machine Learning Repository, Student Performance dataset, DOI 10.24432/C5TG7T.
- **Portuguese:** 649 student rows (recommended).
- **Mathematics:** 395 student rows (alternative).
- **Target:** G3, the final grade (0–20).
- **Potential predictors:** academic, demographic, social, and behavioral features; approximately 30 predictors without G1/G2.
- **No missing values** are reported in the official source, but the pipeline includes imputation to handle future input issues.

### 4. Exploratory analysis

Run `python eda.py --subject por` for `grade_categories.csv`, `descriptive_statistics.csv`, `correlation_matrix.csv`, and `column_quality.csv`. The dashboard also shows class distributions, study-time box plots, grade histograms, interactive relationships, and raw data. Interpret correlations as associations only.

### 5. Preprocessing

1. Stratified development/test split: 80/20.
2. Further fit/validation split within development (approximately 64/16/20 of full data for tuning/test).
3. Median imputation + standardization for numeric columns; mode imputation + one-hot encoding for categorical features.
4. Supervised ANOVA `SelectKBest` fitted on training only (at most 40 encoded columns).
5. Refitting on development data after choosing the hyperparameter configuration, leaving holdout data untouched.

### 6. DNN architecture

```
Raw student details
        ↓
Impute → Scale numeric / One-hot categorical
        ↓
ANOVA SelectKBest (train fitted)
        ↓
Dense hidden layer (ReLU, width tuned)
        ↓
Batch normalization → Dropout
        ↓
Dense hidden layer (ReLU, width/2)
        ↓
Dropout
        ↓
Dense(3, softmax)
        ↓
{Low, Medium, High} model scores
```

- Loss: sparse categorical cross-entropy.
- Optimizer: Adam.
- Regularization: L2 and dropout.
- Early stopping and learning-rate reduction triggered by validation loss.
- Default training: up to 70 epochs; batch size 32; stratified class weights.

### 7. Hyperparameter tuning

When `--tune` is supplied, compare three DNN configurations on the **validation data**:

| Width | Dropout | Learning rate |
|---:|---:|---:|
| 32 | 0.20 | 0.0010 |
| 64 | 0.25 | 0.0010 |
| 128 | 0.35 | 0.0005 |

Choose the configuration by **validation macro-F1**. Early stopping can shorten actual training. The `tuning_results` field of `metrics.json` records the actual experiments and scores.

### 8. Traditional ML benchmark

RandomForestClassifier with 300 trees, `min_samples_leaf=2`, `class_weight='balanced'`, and `max_features='sqrt'`. The model uses the **same selected representation** and **same untouched test split** as the DNN.

### 9. Evaluation and findings (fill after training)

Open `artifacts/por/early_prediction/metrics.json` and record the generated results below:

| Metric (held-out test) | DNN | Random Forest |
|---|---|---|
| Accuracy | **[from metrics.json]** | **[from metrics.json]** |
| Balanced accuracy | **[from metrics.json]** | **[from metrics.json]** |
| Macro F1 | **[from metrics.json]** | **[from metrics.json]** |
| Macro precision | **[from metrics.json]** | **[from metrics.json]** |
| Macro recall | **[from metrics.json]** | **[from metrics.json]** |

Describe the confusion matrices, class imbalance, the tuning-selected architecture, and any signs of overfitting. State which model actually performed better **after** inspecting real results. Include saved images from `artifacts/por/early_prediction/plots/`.

### 10. Streamlit application

Five areas: Overview, Predict, Data Explorer, Model Lab, Methodology. The Predictor accepts a few essential academic inputs and an advanced group of remaining features; it displays a category, all class scores, an appropriate warning, generic study-support suggestions, and an exportable JSON record.

### 11. Limitations and ethics

- Small, historical, single-country secondary-school data; results may not generalize.
- Demographic/family variables can proxy disadvantage or create disparate impacts.
- One holdout split gives a preliminary result. Repeated validation or external datasets would strengthen generalization claims.
- Uncalibrated softmax outputs are not guaranteed probabilities.
- Feature importance is **not** proof of causation.
- Do not use this system for employment, admission, disciplinary, or grading decisions.

### 12. Conclusion

The project offers a complete reproducible demonstration of classification with a DNN and a traditional ML baseline, including preprocessing, EDA, feature selection, tuning, evaluation, and a practical UI. The final conclusion should be based on actual test results recorded after running the training script.

### 13. References

Cortez, P. (2008). *Student Performance*. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T.

Cortez, P., & Silva, A. (2008). Using data mining to predict secondary school student performance. Proceedings of the Future Business Technology Conference.
