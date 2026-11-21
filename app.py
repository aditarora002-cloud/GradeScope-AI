"""GradeScope AI — polished, fully functional Streamlit student-grade predictor."""
from __future__ import annotations

import html
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_utils import (CATEGORY_OPTIONS, CLASSES, DATASET_URL, EDUCATION_LABELS, FEATURE_GROUPS,
                        FEATURE_LABELS, NUMERIC_BOUNDS, NUMERIC_FEATURES, ROOT,
                        STUDY_LABELS, TRAVEL_LABELS, grade_category, input_defaults,
                        load_data, profile_from_example, training_features)
from predict import Predictor, artifact_dir, artifacts_ready

st.set_page_config(page_title="GradeScope AI | Academic Intelligence", page_icon="🎓",
                   layout="wide", initial_sidebar_state="expanded")

CSS_FILE = ROOT / "assets" / "theme.css"
st.markdown(f"<style>{CSS_FILE.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
COLORS = {"Low": "#F7A545", "Medium": "#38BDF8", "High": "#A78BFA"}
PLOTLY_DEFAULTS = dict(
    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, Segoe UI", color="#C3D0E8", size=12),
    margin=dict(l=25, r=25, t=45, b=35),
)


def style_fig(fig: go.Figure, height: int = 355) -> go.Figure:
    fig.update_layout(**PLOTLY_DEFAULTS, height=height,
                      hoverlabel=dict(bgcolor="#1F2A41", font_color="#FFF"),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    fig.update_xaxes(showgrid=False, linecolor="#35415B", tickfont_color="#99A8C1")
    fig.update_yaxes(gridcolor="#22314A", zeroline=False, tickfont_color="#99A8C1")
    return fig


def chart(fig, height=355):
    st.plotly_chart(style_fig(fig, height), use_container_width=True,
                    config={"displayModeBar": False, "responsive": True})


def hero(eyebrow: str, title: str, subtitle: str, emoji="✦"):
    st.markdown(
        '<div class="hero"><div class="eyebrow">' + html.escape(eyebrow) + '</div>'
        '<h1>' + html.escape(title) + '</h1><p>' + html.escape(subtitle) + '</p>'
        '<span class="orb" aria-hidden="true">' + html.escape(emoji) + '</span></div>',
        unsafe_allow_html=True,
    )


def stat_card(title: str, value: str, sub: str):
    st.markdown('<div class="stat-card"><div class="tag">' + html.escape(title) +
                '</div><div class="value">' + html.escape(value) +
                '</div><div class="footer">' + html.escape(sub) + '</div></div>', unsafe_allow_html=True)


def section(name: str):
    st.markdown(f'<div class="title-row">{html.escape(name)}</div>', unsafe_allow_html=True)


def note(message: str):
    st.markdown(f'<div class="note-card">{html.escape(message)}</div>', unsafe_allow_html=True)


@st.cache_data(show_spinner=False, ttl=3600)
def cached_df(subject: str):
    return load_data(subject)


@st.cache_resource(show_spinner=False)
def cached_predictor(path_string: str, metadata_modified: float):
    del metadata_modified
    return Predictor(Path(path_string))


def get_predictor(subject: str, with_grades: bool):
    if not artifacts_ready(subject, with_grades):
        return None
    directory = artifact_dir(subject, with_grades)
    return cached_predictor(str(directory), (directory / "metadata.json").stat().st_mtime)


def safe_df(subject: str):
    try:
        return cached_df(subject), None
    except Exception as exc:
        return None, str(exc)


def grade_plot(df: pd.DataFrame):
    cats = grade_category(df["G3"])
    count = cats.value_counts().reindex(CLASSES, fill_value=0).rename_axis("Grade").reset_index(name="Students")
    fig = px.bar(count, x="Grade", y="Students", color="Grade", text="Students",
                 color_discrete_map=COLORS, title="Performance category distribution")
    fig.update_traces(marker_line_width=0, textposition="outside", width=.58,
                      hovertemplate="%{x}<br>%{y} students<extra></extra>")
    fig.update_layout(showlegend=False, yaxis_title="Students", xaxis_title="")
    return fig


def grade_histogram(df: pd.DataFrame):
    fig = px.histogram(df, x="G3", nbins=21, color_discrete_sequence=["#A78BFA"],
                       title="Final grade distribution", labels={"G3": "Final grade (out of 20)"})
    fig.update_traces(marker_line_color="#1A2541", marker_line_width=.75)
    fig.update_layout(bargap=.08, yaxis_title="Students", xaxis=dict(dtick=2))
    return fig


def study_box(df: pd.DataFrame):
    data = df.copy()
    data["Weekly study time"] = data["studytime"].map(STUDY_LABELS)
    fig = px.box(data, x="Weekly study time", y="G3", color="Weekly study time",
                 title="Study time and final grades", points="outliers",
                 category_orders={"Weekly study time": list(STUDY_LABELS.values())},
                 color_discrete_sequence=["#F7A545", "#38BDF8", "#A78BFA", "#34D399"])
    fig.update_layout(showlegend=False, yaxis_title="Final grade / 20", xaxis_title="")
    return fig


def train_panel(subject: str, with_grades: bool):
    st.warning("No trained model for this dataset and stage yet. Train once to activate the predictor and Model Lab.")
    args_line = f"python train.py --subject {subject} --tune" + (" --include-prior-grades" if with_grades else "")
    st.code(args_line, language="powershell")
    col1, col2 = st.columns([1, 1])
    with col1:
        auto_train = st.button("⚡ Train model inside the app", type="primary", use_container_width=True)
    with col2:
        st.caption("Training uses TensorFlow on your computer and can take several minutes on CPU.")
    if auto_train:
        params = [sys.executable, str(ROOT / "train.py"), "--subject", subject,
                  "--epochs", "55", "--tune"]
        if with_grades:
            params.append("--include-prior-grades")
        try:
            with st.status("Training DNN and Random Forest", state="running", expanded=True) as status:
                process = subprocess.Popen(
                    params, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding="utf-8", errors="replace", bufsize=1,
                )
                recent_lines = []
                assert process.stdout is not None
                progress_area = st.empty()
                for line in process.stdout:
                    recent_lines.append(line.rstrip())
                    progress_area.code("\n".join(recent_lines[-14:]), language="text")
                code = process.wait()
                if code != 0:
                    status.update(label="Training failed. Check the log above.", state="error", expanded=True)
                    st.error(f"Training exited with code {code}. See the detailed log above.")
                else:
                    status.update(label="Models trained successfully", state="complete", expanded=False)
                    st.cache_resource.clear()
                    st.success("Models saved! Reloading dashboard.")
                    st.rerun()
        except Exception as exc:
            st.exception(exc)


def overview(df: pd.DataFrame | None, subject: str, predictor):
    hero("STUDENT INTELLIGENCE PLATFORM  /  V2.0", "Make academic insights feel effortless.",
         "Explore real student performance patterns, compare machine learning models, "
         "and generate transparent academic category estimates — all in one research dashboard.", "✧")
    if df is None:
        st.error("The UCI dataset isn't available. Connect to the internet for the first download, or place student-por.csv / student-mat.csv inside the data folder.")
        return
    cats = grade_category(df["G3"])
    a, b, c, d = st.columns(4)
    with a: stat_card("STUDENT RECORDS", f"{len(df):,}", "Official UCI sample")
    with b: stat_card("AVERAGE FINAL GRADE", f"{df['G3'].mean():.1f} / 20", "Descriptive, not predicted")
    with c: stat_card("HIGH PERFORMANCE", f"{(cats == 'High').mean()*100:.1f}%", "Final grade ≥ 15")
    with d: stat_card("MODEL STATUS", "Ready" if predictor else "Not trained", "DNN + Random Forest")
    st.markdown('<div class="section-label">DATA AT A GLANCE</div>', unsafe_allow_html=True)
    left, right = st.columns(2, gap="medium")
    with left: chart(grade_plot(df))
    with right: chart(grade_histogram(df))
    section("Learn from the data")
    l, r = st.columns([1.1, .9], gap="medium")
    with l: chart(study_box(df))
    with r:
        st.markdown("### What this project predicts")
        st.markdown("Each prediction is a **category of final academic grade (G3)**. "
                    "The model processes course, household and behavioral information using "
                    "a DNN and a Random Forest baseline.")
        st.markdown('<span class="tag-pill amber">LOW · 0–9</span>'
                    '<span class="tag-pill aqua">MEDIUM · 10–14</span>'
                    '<span class="tag-pill purple">HIGH · 15–20</span>', unsafe_allow_html=True)
        st.markdown("\n\n**Early estimation by default** — avoids G1/G2, which are only available "
                    "after earlier grading periods and make the task considerably easier.")
        st.info("This is a classroom demonstration using historical Portuguese secondary-school data. "
                "Do not use model predictions to judge employability, admission, or individuals.")
    st.markdown('<div class="footerline">Source: UCI Student Performance (Cortez & Silva, 2008) · '
                '<a href="https://archive.ics.uci.edu/dataset/320/student+performance">View the official dataset</a></div>',
                unsafe_allow_html=True)


def input_widget(field: str, default, prefix: str):
    label = FEATURE_LABELS.get(field, field)
    key = f"{prefix}-{field}"
    if field in CATEGORY_OPTIONS:
        choices = list(CATEGORY_OPTIONS[field])
        initial = choices.index(str(default)) if str(default) in choices else 0
        return st.selectbox(label, choices, index=initial, format_func=lambda k: CATEGORY_OPTIONS[field][k], key=key)
    if field in NUMERIC_BOUNDS:
        low, high = NUMERIC_BOUNDS[field]
        default = max(low, min(high, int(default)))
        if field == "studytime":
            return st.selectbox(label, list(STUDY_LABELS), index=default-1,
                                format_func=lambda i: STUDY_LABELS[i], key=key)
        if field == "traveltime":
            return st.selectbox(label, list(TRAVEL_LABELS), index=default-1,
                                format_func=lambda i: TRAVEL_LABELS[i], key=key)
        if field in ("Medu", "Fedu"):
            return st.selectbox(label, list(EDUCATION_LABELS), index=default,
                                format_func=lambda i: EDUCATION_LABELS[i], key=key)
        if field in ("age", "absences", "failures", "G1", "G2"):
            return st.number_input(label, min_value=low, max_value=high, value=default, step=1, key=key)
        return st.slider(label, min_value=low, max_value=high, value=default, step=1, key=key)
    # Fallback for any unknown but legitimate categorical feature.
    return st.text_input(label, value=str(default), key=key)


def suggested_focus(profile: dict) -> list[str]:
    """Generic educational prompts; never claimed as causal DNN explanations."""
    tips = []
    if profile.get("studytime", 3) <= 2:
        tips.append("Consider a more regular weekly study schedule, adjusted to your learning needs.")
    if profile.get("absences", 0) >= 10:
        tips.append("Check whether attendance barriers can be reduced with school support.")
    if profile.get("failures", 0) >= 1:
        tips.append("Consider additional revision, tutoring, or targeted support in difficult topics.")
    if profile.get("schoolsup") == "no":
        tips.append("Explore available school learning-support resources if needed.")
    if not tips:
        tips.append("Maintain a consistent routine and review progress regularly with educators.")
    return tips


def prediction_page(subject: str, with_grades: bool, predictor, df: pd.DataFrame | None):
    hero("PREDICTION WORKSPACE", "One student profile. Two intelligent models.",
         "Fill out a short academic snapshot, refine the optional details, and see a "
         "category estimate alongside all three model probabilities.", "◎")
    if predictor is None:
        train_panel(subject, with_grades)
        note("You'll still be able to use the Explorer before training, provided the dataset has been downloaded.")
        return
    defaults = predictor.metadata["input_defaults"]
    fields = predictor.metadata["feature_columns"]
    a, b = st.columns([1, 1])
    with a:
        scenario = st.selectbox("Starting profile", ["Typical profile", "Consistent learner", "Needs support"],
                                help="Illustrative scenarios, not real student records")
    with b:
        model_name = st.selectbox("Model to use", ["DNN", "Random Forest"])
    starting = profile_from_example(defaults, scenario)
    st.markdown('<div class="model-ribbon">'
                f'{len(fields)} supported fields · {"Prior grades included" if with_grades else "Early prediction: G1/G2 excluded"} '
                '· Inputs you do not edit use reference median/mode values from the dataset</div>',
                unsafe_allow_html=True)
    primary = ["studytime", "failures", "absences", "schoolsup", "higher", "age", "internet", "famrel"]
    primary = [field for field in primary if field in fields]
    profile = dict(starting)
    prefix = f"{subject}-{'prior' if with_grades else 'early'}-{scenario}"
    with st.form("prediction_form"):
        section("Academic snapshot")
        col1, col2 = st.columns(2, gap="large")
        for i, field in enumerate(primary):
            with (col1 if i % 2 == 0 else col2):
                profile[field] = input_widget(field, starting[field], prefix)
        with st.expander("Advanced inputs · personalize all remaining fields"):
            st.caption("Some fields are sensitive. They reproduce the original dataset for coursework only; "
                       "avoid using them for decisions about real individuals.")
            others = [field for field in fields if field not in primary]
            for group, group_fields in FEATURE_GROUPS.items():
                members = [f for f in group_fields if f in others]
                if not members:
                    continue
                st.markdown(f"**{group}**")
                c1, c2 = st.columns(2)
                for i, field in enumerate(members):
                    with (c1 if i % 2 == 0 else c2):
                        profile[field] = input_widget(field, starting[field], prefix)
        submitted = st.form_submit_button("✦  Generate academic prediction", type="primary", use_container_width=True)
    if submitted:
        try:
            prediction = predictor.predict(profile, model_name=model_name)
            st.session_state["latest_prediction"] = {"prediction": prediction, "subject": subject,
                                                      "with_grades": with_grades}
        except Exception as exc:
            st.error(f"Prediction error: {exc}")
    result_state = st.session_state.get("latest_prediction")
    if not result_state or result_state["subject"] != subject or result_state["with_grades"] != with_grades:
        note("Select a scenario or enter your own information, then click Generate academic prediction to see the model result.")
        return
    result = result_state["prediction"]
    section("Prediction output")
    col1, col2 = st.columns([.82, 1.18], gap="medium")
    with col1:
        label = result["category"]
        st.markdown('<div class="prediction-card"><div class="kicker">PREDICTED CATEGORY</div>'
                    f'<div class="big">{html.escape(label)}</div>'
                    f'<div class="subtitle">{label} · { {"Low":"0–9", "Medium":"10–14", "High":"15–20"}[label]} out of 20</div>'
                    f'<span class="tag-pill purple">{html.escape(result["model"])} model</span>'
                    f'<span class="tag-pill">{result["confidence"]:.1%} top-class score</span>'
                    '</div>', unsafe_allow_html=True)
        st.caption("The percentage is the model's uncalibrated score, not a guaranteed chance of a grade outcome.")
    with col2:
        probs = result["probabilities"]
        fig = go.Figure(go.Bar(
            x=[probs[k] * 100 for k in CLASSES], y=CLASSES, orientation="h",
            marker_color=[COLORS[k] for k in CLASSES],
            text=[f"{probs[k]*100:.1f}%" for k in CLASSES], textposition="outside",
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
        ))
        fig.update_layout(title="Model scores across categories", xaxis_title="Score (%)",
                          yaxis=dict(autorange="reversed"), xaxis=dict(range=[0, 112]))
        chart(fig, height=310)
    l, r = st.columns([1.2, .8], gap="large")
    with l:
        st.markdown("#### Supportive next steps")
        for tip in suggested_focus(result["input"]):
            st.markdown(f"• {tip}")
        st.caption("These are generic educational suggestions based on form inputs, not explanations of the model's internal reasoning.")
    with r:
        payload = {"model": result["model"], "subject": subject,
                   "using_prior_grades": with_grades, "prediction": result["category"],
                   "model_scores": result["probabilities"], "input": result["input"],
                   "warning": "Educational demo; not an employability evaluation."}
        st.download_button("↓ Export prediction JSON", data=json.dumps(payload, indent=2),
                           file_name="gradescope_prediction.json", mime="application/json", use_container_width=True)
        st.caption("Download contains the full entered profile; treat it as personal data if used with real details.")
    note("A small historical sample is insufficient to establish reliability across different schools, countries or time periods. "
         "Do not use this output for grading, admissions, hiring or other consequential decisions.")


def explorer(df: pd.DataFrame | None, subject: str):
    hero("DATA EXPLORER", "Discover the story inside the dataset.",
         "Interactive exploratory analysis of the original UCI records: class balance, "
         "individual attributes, correlations, and the underlying data table.", "◈")
    if df is None:
        st.error("Data download failed. Place the official student CSV in the data directory and refresh.")
        return
    a, b, c, d = st.columns(4)
    with a: stat_card("DATASET", "Portuguese" if subject == "por" else "Mathematics", "UCI / Cortez & Silva")
    with b: stat_card("SAMPLES", f"{len(df):,}", "Real historical records")
    with c: stat_card("INPUT VARIABLES", "32", "30 + G1 + G2")
    with d: stat_card("MISSING CELLS", f"{df.isna().sum().sum():,}", "Raw file quality")
    st.markdown('<div class="section-label">EXPLORE DISTRIBUTIONS</div>', unsafe_allow_html=True)
    l, r = st.columns(2, gap="medium")
    with l: chart(grade_plot(df))
    with r: chart(study_box(df))
    section("Explore a relationship")
    left, right = st.columns([.34, .66])
    with left:
        numeric_cols = [col for col in ["absences", "studytime", "failures", "age", "freetime", "goout", "health", "G1", "G2"] if col in df]
        selected = st.selectbox("Choose an attribute", numeric_cols)
        st.caption("Explore association, not proof of cause and effect.")
    with right:
        frame = df.assign(category=grade_category(df["G3"]))
        fig = px.scatter(frame, x=selected, y="G3", color="category", opacity=.62,
                         color_discrete_map=COLORS, category_orders={"category": CLASSES},
                         title=f"{FEATURE_LABELS.get(selected, selected)} vs. final grade",
                         labels={"G3": "Final grade / 20", "category": "Grade category"})
        fig.update_traces(marker=dict(size=7, line=dict(width=0)))
        chart(fig, height=360)
    section("Numeric feature correlations")
    corr = df.select_dtypes(include="number").corr(numeric_only=True).round(2)
    corr_fig = px.imshow(corr, zmin=-1, zmax=1, color_continuous_scale="RdBu_r",
                         title="Pearson correlation matrix", aspect="auto")
    corr_fig.update_layout(coloraxis_colorbar=dict(title="r"))
    chart(corr_fig, height=555)
    section("Original data viewer")
    with st.expander("Show all records with filters", expanded=False):
        categories = st.multiselect("Show grades", CLASSES, default=CLASSES)
        show = df.copy()
        show.insert(0, "Category", grade_category(show["G3"]).to_numpy())
        filtered = show[show["Category"].isin(categories)]
        st.dataframe(filtered, use_container_width=True, height=400, hide_index=True)
        st.download_button("↓ Download filtered data (CSV)", filtered.to_csv(index=False).encode("utf-8"),
                           file_name=f"uci_{subject}_filtered.csv", mime="text/csv")
    note("EDA includes all rows for descriptive research. Training transforms, feature selection, "
         "and DNN tuning use only the development data; final test data is untouched until evaluation.")


def confusion_chart(metrics: dict, model: str):
    cm = np.array(metrics["models"][model]["confusion_matrix"])
    fig = go.Figure(go.Heatmap(z=cm, x=CLASSES, y=CLASSES, colorscale=[[0, "#101B30"], [.45, "#4769BD"], [1, "#A78BFA"]],
                               showscale=False, text=cm.astype(str), texttemplate="%{text}",
                               textfont=dict(size=18, color="white"), hovertemplate="Actual %{y}, Predicted %{x}: %{z}<extra></extra>"))
    fig.update_layout(title=f"{model} · held-out confusion matrix", xaxis_title="Predicted", yaxis_title="Actual",
                      yaxis=dict(autorange="reversed"))
    return fig


def model_lab(subject: str, with_grades: bool, predictor):
    hero("MODEL LABORATORY", "Audit the models. Don't just trust a score.",
         "Inspect holdout evaluation, confusion matrices, learning curves, feature selection, "
         "and tuning results from the models actually trained on your computer.", "⌁")
    if predictor is None:
        train_panel(subject, with_grades)
        return
    meta = predictor.metadata
    metrics = predictor.metrics
    st.markdown('<span class="tag-pill purple">HELD-OUT TEST DATA</span>'
                '<span class="tag-pill aqua">DNN VS RANDOM FOREST</span>', unsafe_allow_html=True)
    st.caption(f"Model trained: {meta['trained_at_utc'][:10]} UTC · "
               f"{meta['train_count']} fit / {meta['validation_count']} tuning validation / "
               f"{meta['test_count']} test samples · {len(meta['selected_features'])} selected encoded features")
    comparison = []
    for model in ("DNN", "Random Forest"):
        report = metrics["models"][model]
        comparison.append({"Model": model,
                           "Accuracy": round(report["accuracy"], 4),
                           "Balanced accuracy": round(report["balanced_accuracy"], 4),
                           "Macro F1": round(report["macro_f1"], 4),
                           "Macro precision": round(report["macro_precision"], 4),
                           "Macro recall": round(report["macro_recall"], 4)})
    rows = pd.DataFrame(comparison)
    section("Holdout performance")
    k1, k2, k3, k4 = st.columns(4)
    dnn = metrics["models"]["DNN"]
    rf = metrics["models"]["Random Forest"]
    with k1: stat_card("DNN ACCURACY", f"{dnn['accuracy']:.1%}", "Unseen test split")
    with k2: stat_card("DNN MACRO F1", f"{dnn['macro_f1']:.3f}", "Equal weight across classes")
    with k3: stat_card("RF ACCURACY", f"{rf['accuracy']:.1%}", "Baseline comparison")
    with k4: stat_card("RF MACRO F1", f"{rf['macro_f1']:.3f}", "Equal weight across classes")
    st.dataframe(rows, hide_index=True, use_container_width=True)
    plot_data = rows.melt(id_vars="Model", value_vars=["Accuracy", "Balanced accuracy", "Macro F1"],
                          var_name="Metric", value_name="Score")
    fig = px.bar(plot_data, x="Metric", y="Score", color="Model", barmode="group", text_auto=".3f",
                 color_discrete_map={"DNN": "#A78BFA", "Random Forest": "#38BDF8"}, title="Head-to-head evaluation")
    fig.update_layout(yaxis=dict(range=[0, 1.08]))
    chart(fig, height=350)
    l, r = st.columns(2)
    with l: chart(confusion_chart(metrics, "DNN"))
    with r: chart(confusion_chart(metrics, "Random Forest"))
    section("DNN learning dynamics")
    history = metrics.get("training_history", {})
    if history.get("loss"):
        epoch = np.arange(1, len(history["loss"])+1)
        history_fig = go.Figure()
        for name, color in [("loss", "#A78BFA"), ("val_loss", "#38BDF8")]:
            if name in history:
                history_fig.add_trace(go.Scatter(x=epoch, y=history[name], mode="lines", name=name,
                                                 line=dict(color=color, width=3)))
        history_fig.update_layout(title="Training / validation loss", xaxis_title="Epoch", yaxis_title="Cross-entropy")
        chart(history_fig)
    section("Selected feature importance")
    importance_file = predictor.directory / "feature_importance.csv"
    if importance_file.exists():
        importance = pd.read_csv(importance_file).head(18).sort_values("importance")
        fig = px.bar(importance, x="importance", y="feature", orientation="h", title="Random Forest importance · top 18",
                     color_discrete_sequence=["#8B5CF6"])
        fig.update_layout(yaxis_title="", xaxis_title="Feature importance")
        chart(fig, height=520)
        st.caption("The Random Forest importance chart is not a causal explanation and does not describe the DNN's inner reasoning.")
    with st.expander("DNN hyperparameter experiments and architecture", expanded=False):
        st.dataframe(pd.DataFrame(metrics.get("tuning_results", [])), hide_index=True, use_container_width=True)
        st.json(meta["best_dnn_config"])
        st.markdown("**Architecture:** input → dense ReLU → batch normalization → dropout → "
                    "dense ReLU → dropout → softmax (3 classes). Early stopping and validation-based "
                    "hyperparameter tuning reduce overfitting.")
    st.download_button("↓ Export complete evaluation (JSON)", json.dumps(metrics, indent=2),
                       file_name=f"gradescope_{subject}_evaluation.json", mime="application/json")
    note("One holdout split gives an initial benchmark, not a proof that a DNN outperforms simpler models. "
         "Repeat experiments with stratified cross-validation for stronger claims.")


def methodology(subject: str, with_grades: bool):
    hero("DOCUMENTATION / RESEARCH", "Built to demonstrate the whole ML lifecycle.",
         "From UCI dataset preparation through deep learning, feature selection, "
         "hyperparameter tuning, test evaluation, and deployment.", "☷")
    st.markdown("### Experiment design")
    for number, title, copy in [
        ("01", "Data sourcing & cleaning", "Download the 649-row Portuguese or 395-row Mathematics UCI dataset; validate all expected attributes, inspect missingness, and construct three grade categories."),
        ("02", "Leakage-aware split", "Stratified 80% development / 20% untouched test split, then a validation split within development for architecture selection."),
        ("03", "Preprocessing & feature selection", "Train-only median/mode imputation, numeric standardization, one-hot categorical encoding, and ANOVA SelectKBest."),
        ("04", "Deep Neural Network", "A two-hidden-layer Keras classifier with ReLU, batch normalization, dropout, L2 penalty, Adam, softmax, early stopping and tuning."),
        ("05", "Traditional ML benchmark", "RandomForestClassifier evaluated on the exact same untouched test records, with accuracy, balanced accuracy, macro F1 and class confusion matrices."),
        ("06", "Interactive deployment", "Saved inference pipeline serves the Streamlit Predictor; Explorer and Model Lab expose underlying data and actual trained metrics."),
    ]:
        st.markdown(f"**{number} — {title}**")
        st.caption(copy)
    st.markdown("### Practical commands")
    suffix = " --include-prior-grades" if with_grades else ""
    st.code(f"python train.py --subject {subject} --tune{suffix}\n"
            f"python eda.py --subject {subject}\n"
            "python -m streamlit run app.py", language="powershell")
    st.markdown("### Study assumptions & limitations")
    st.markdown("The UCI study describes Portuguese secondary-school pupils. "
                "**Low = G3 0–9, Medium = 10–14, High = 15–20** are project-defined, "
                "not validated educational or employment thresholds. Later-year grade mode "
                "uses G1/G2 and will generally be easier, but cannot be used before those grades exist.")
    st.markdown("No model prediction should be used to decide employment, admissions, discipline, "
                "eligibility, or student worth. Demographic and household features can encode "
                "disadvantage; results are illustrative and require fairness validation before any real-world application.")
    st.markdown("### Original citation")
    st.markdown("Cortez, P. (2008). *Student Performance*. UCI Machine Learning Repository. "
                "https://doi.org/10.24432/C5TG7T. Dataset license: CC BY 4.0.")
    st.link_button("↗ Open official UCI dataset", DATASET_URL)


# --- Main navigation ---
with st.sidebar:
    st.markdown('<div class="sidebar-logo"><span class="logo-mark">✦</span> GradeScope <span style="color:#A78BFA">AI</span></div>',
                unsafe_allow_html=True)
    st.caption("ACADEMIC INTELLIGENCE / STUDENT PERFORMANCE")
    st.markdown('<div class="mini-title">WORKSPACE</div>', unsafe_allow_html=True)
    nav = st.radio("Go to", ["⌂  Overview", "✦  Predict", "◈  Data Explorer", "≋  Model Lab", "☷  Methodology"],
                   label_visibility="collapsed", key="nav")
    st.markdown('<div class="mini-title">EXPERIMENT SETTINGS</div>', unsafe_allow_html=True)
    subject_name = st.selectbox("Dataset subject", ["Portuguese · 649 records", "Mathematics · 395 records"])
    subject = "por" if subject_name.startswith("Portuguese") else "mat"
    with_grades = st.toggle("Include prior grades (G1, G2)", value=False,
                            help="Leave off for earlier-year predictions; enabling makes it a later-year task.")
    st.markdown('<div class="mini-title">STATUS</div>', unsafe_allow_html=True)
    ready = artifacts_ready(subject, with_grades)
    st.markdown('<span class="tag-pill aqua">● TRAINED MODEL READY</span>' if ready else
                '<span class="tag-pill amber">● TRAINING REQUIRED</span>', unsafe_allow_html=True)
    st.caption("Training files are saved separately for each subject and grade mode.")
    st.markdown('<div class="footerline">GradeScope AI v2.0<br>UCI Student Performance · Educational demo</div>',
                unsafe_allow_html=True)

df, df_error = safe_df(subject)
try:
    pred = get_predictor(subject, with_grades)
except Exception as exc:
    pred = None
    st.error(f"Saved model could not be loaded: {exc}. If you changed dependencies, retrain the model.")

if df_error and nav != "✦  Predict":
    st.caption("Dataset status: unable to download UCI records; see the Data Explorer or README for an offline installation.")
if nav == "⌂  Overview":
    overview(df, subject, pred)
elif nav == "✦  Predict":
    prediction_page(subject, with_grades, pred, df)
elif nav == "◈  Data Explorer":
    explorer(df, subject)
elif nav == "≋  Model Lab":
    model_lab(subject, with_grades, pred)
else:
    methodology(subject, with_grades)
