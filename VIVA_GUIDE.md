# GradeScope AI — Evaluation / Viva Guide

## 60-second explanation

"GradeScope AI is a deep learning-based project for classifying students' final academic performance using the official UCI Student Performance dataset. I use the Portuguese dataset by default, with 649 records. The target is final grade G3, converted into three categories: Low (0–9), Medium (10–14), and High (15–20). The system performs preprocessing, encoding, scaling, ANOVA feature selection, DNN training and validation-based hyperparameter tuning. I compare the trained DNN against a Random Forest on the same unseen test split, and present the results in a Streamlit dashboard. The dashboard also contains an EDA explorer and interactive prediction form. The goal is an educational demonstration of a complete machine-learning pipeline, not an actual employability assessment."

## Questions examiners may ask

**1. Why use a DNN?**

It can model nonlinear relationships among selected encoded attributes. However, 649 rows is small for deep learning, so performance is compared against a simpler Random Forest rather than assumed superior.

**2. What is the target variable?**

`G3` — final grade out of 20. The project converts it into Low, Medium, and High categories for multi-class classification.

**3. Why not predict exact grades directly?**

The assignment asks for performance categories. A regression version could predict G3 directly, but would require regression loss/metrics and grade-to-band conversion at inference.

**4. How are strings handled?**

One-hot encoding for categorical features. Numerical features are median-imputed and standardized.

**5. Why do you select features?**

The transformed feature matrix may contain many one-hot columns. ANOVA `SelectKBest` selects up to 40 columns associated with training labels, reducing dimensionality. The selector never fits to the test set.

**6. What is a train/validation/test split?**

Training is used to fit weights and transforms; validation is for architecture selection and early stopping; the independent test split assesses generalization.

**7. Why exclude G1 and G2 by default?**

These are first- and second-period grades strongly correlated with G3. Including them can improve accuracy but changes the practical question to predicting after earlier grades are available. The default model tests earlier detection without these predictors.

**8. Which optimizer and loss do you use?**

Adam and sparse categorical cross-entropy with a three-class softmax output.

**9. What techniques reduce overfitting?**

Dropout, L2 weight regularization, early stopping, learning-rate reduction, and a validation-based architecture search.

**10. Why do you use macro-F1?**

Macro-F1 gives each class equal weight, which is useful if Low / Medium / High have different sample counts. Accuracy alone can hide weak minority-class performance.

**11. Why Random Forest?**

It is a strong, traditional non-linear classifier and provides a reasonable benchmark for small tabular datasets.

**12. What does the confusion matrix show?**

Rows are actual classes, columns predicted classes, and cells count the classification decisions. The diagonal contains correct predictions.

**13. How do you know which model performed better?**

**Do not invent a result.** Open the Model Lab after training and compare the actual test macro-F1 and accuracy, including errors by class.

**14. Why can the softmax score not be treated as a true probability?**

Neural-network class scores are not necessarily calibrated; calibration testing is required before interpreting scores as frequencies of outcomes.

**15. Can you use this for hiring or employability?**

No. The dataset describes grades from Portuguese secondary schools, not workplace ability. Using it to make admissions or employment decisions would be inappropriate and unsupported.

## 10-day plan

| Day | Milestone |
|---|---|
| 1 | Understand UCI source, target, feature documentation |
| 2 | Data loading, validation, preprocessing |
| 3 | Exploratory analysis and correlation plots |
| 4 | Label construction and feature selection |
| 5 | DNN architecture and first training |
| 6 | Hyperparameter tuning and regularization |
| 7 | Train Random Forest baseline; inspect held-out metrics |
| 8 | Implement Streamlit inputs and inference pipeline |
| 9 | Dashboard design, tests, plots, bug fixing |
| 10 | Report, demonstrations, evaluation and slides |

## Demo sequence (under 3 minutes)

1. Open Overview and explain grade categories and dataset size.
2. Switch to Data Explorer; show histogram, study-time box plot, and correlation matrix.
3. Open Model Lab; demonstrate actual DNN and Random Forest accuracy, macro-F1 and confusion matrices.
4. Open Predict; select a sample profile, change study time/absences, and generate a prediction.
5. Open Methodology; explain train-only preprocessing, tuning, saved inference, and limitations.
