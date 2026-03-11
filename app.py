"""
🏙️ Chicago Crime ML Pipeline — Streamlit App
Supports: Arrest Prediction · Crime Type Prediction · Crime Count per Area
"""

import sys, types, json, warnings
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
from packaging.version import Version
from pathlib import Path

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Chicago Crime ML",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# SKLEARN COMPATIBILITY SHIM
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def _install_shims():
    import sklearn
    _skv = Version(sklearn.__version__)
    if _skv >= Version("1.3"):
        from sklearn._loss import loss as _loss_src
        _NAME_MAP = {
            "BinaryCrossEntropy":      "HalfBinomialLoss",
            "CategoricalCrossEntropy": "HalfMultinomialLoss",
            "LeastSquares":            "HalfSquaredError",
            "LeastAbsoluteDeviation":  "AbsoluteError",
            "Poisson":                 "HalfPoissonLoss",
            "Gamma":                   "HalfGammaLoss",
            "Quantile":                "PinballLoss",
        }
        def _make_shim(fake_path, real_module, name_map=None):
            shim = types.ModuleType(fake_path)
            for name in dir(real_module):
                setattr(shim, name, getattr(real_module, name))
            if name_map:
                for old, new in name_map.items():
                    if hasattr(real_module, new):
                        setattr(shim, old, getattr(real_module, new))
            sys.modules[fake_path] = shim
        _make_shim("sklearn.ensemble._hist_gradient_boosting.loss",  _loss_src, _NAME_MAP)
        _make_shim("sklearn.ensemble._hist_gradient_boosting._loss", _loss_src, _NAME_MAP)
    elif _skv >= Version("1.0"):
        from sklearn.ensemble._hist_gradient_boosting import _loss as _loss_src
        shim = types.ModuleType("sklearn.ensemble._hist_gradient_boosting.loss")
        for name in dir(_loss_src):
            setattr(shim, name, getattr(_loss_src, name))
        sys.modules["sklearn.ensemble._hist_gradient_boosting.loss"] = shim

_install_shims()

DATA_DIR  = Path("ml_data")
MODEL_DIR = Path("ml_models")

# ══════════════════════════════════════════════════════════════════════════════
# LOAD MODELS
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_models():
    import sklearn
    from packaging.version import Version as _V
    _skv = _V(sklearn.__version__)
    m1 = joblib.load(MODEL_DIR / "task1_best_model.pkl")
    m2 = joblib.load(MODEL_DIR / "task2_best_model.pkl")
    m3 = joblib.load(MODEL_DIR / "task3_best_model.pkl")
    if _skv >= _V("1.3"):
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
        from sklearn._loss import loss as _loss_module
        _LOSS_REMAP = {
            "LeastSquares":            "HalfSquaredError",
            "BinaryCrossEntropy":      "HalfBinomialLoss",
            "CategoricalCrossEntropy": "HalfMultinomialLoss",
            "LeastAbsoluteDeviation":  "AbsoluteError",
            "Poisson":                 "HalfPoissonLoss",
            "Gamma":                   "HalfGammaLoss",
            "Quantile":                "PinballLoss",
        }
        def _patch(model):
            if not isinstance(model, (HistGradientBoostingClassifier, HistGradientBoostingRegressor)):
                return model
            for attr, default in [("_preprocessor", None), ("_is_fitted", True), ("_no_validation", False)]:
                if not hasattr(model, attr):
                    setattr(model, attr, default)
            if hasattr(model, "_baseline_prediction"):
                bp = model._baseline_prediction
                if bp.ndim == 2 and bp.shape[1] == 1 and bp.shape[0] > 1:
                    model._baseline_prediction = bp.T
            if hasattr(model, "_loss"):
                old_name = type(model._loss).__name__
                new_name = _LOSS_REMAP.get(old_name, old_name)
                if hasattr(_loss_module, new_name):
                    try:
                        new_cls = getattr(_loss_module, new_name)
                        n_classes = getattr(model._loss, "n_classes", None)
                        model._loss = new_cls(n_classes=n_classes) if n_classes else new_cls()
                    except Exception:
                        pass
            return model
        m1, m2, m3 = _patch(m1), _patch(m2), _patch(m3)
    return m1, m2, m3

@st.cache_resource
def load_meta():
    metas, scalers = {}, {}
    for task in ["task1", "task2", "task3"]:
        with open(DATA_DIR / f"{task}_meta.json") as f:
            metas[task] = json.load(f)
        scaler_path = DATA_DIR / f"{task}_scaler.pkl"
        scalers[task] = joblib.load(scaler_path) if scaler_path.exists() else None
    with open(DATA_DIR / "task2_label_map.json") as f:
        label_map = json.load(f)
    with open(MODEL_DIR / "training_summary.json") as f:
        train_summary = json.load(f)
    scorecard = pd.read_csv(DATA_DIR / "05_scorecard.csv")
    return metas, scalers, label_map, train_summary, scorecard

@st.cache_resource
def load_test_data():
    arrays = {}
    for task in ["task1", "task2", "task3"]:
        data = np.load(DATA_DIR / f"{task}_preprocessed.npz", allow_pickle=True)
        arrays[task] = {"X_test": data["X_test"], "y_test": data["y_test"]}
    return arrays

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE VECTOR BUILDER
# ══════════════════════════════════════════════════════════════════════════════
CYCLICAL_BASES = {"Hour": 24, "Month": 12, "DayOfWeek": 7}

def build_input_vector(feature_names, input_vals):
    row = []
    for feat in feature_names:
        if feat in input_vals:
            row.append(float(input_vals[feat]))
        elif feat.endswith("_sin") or feat.endswith("_cos"):
            base = feat.rsplit("_", 1)[0]
            max_val = CYCLICAL_BASES.get(base, 12)
            raw = input_vals.get(base, 0)
            val = (np.sin if feat.endswith("_sin") else np.cos)(2 * np.pi * raw / max_val)
            row.append(float(val))
        elif feat.startswith("Season_"):
            season_name = feat.split("Season_")[1]
            row.append(1.0 if input_vals.get("Season") == season_name else 0.0)
        elif feat == "Night_x_Weekend":
            row.append(float(input_vals.get("IsNight", 0)) * float(input_vals.get("IsWeekend", 0)))
        elif feat == "Domestic_x_Night":
            row.append(float(input_vals.get("Domestic", 0)) * float(input_vals.get("IsNight", 0)))
        else:
            row.append(0.0)
    return np.array(row, dtype=float).reshape(1, -1)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("🏙️ Chicago Crime ML")
    st.caption("Machine learning on Chicago crime data")
    st.divider()
    st.markdown("""
### 🎯 Prediction Tasks

🔵 **Arrest Prediction**
Will this crime result in an arrest?

🟢 **Crime Type Prediction**
What category of crime is this?

🟣 **Crime Count per Area**
How many crimes will occur in this community area?
    """)
    st.divider()
    st.caption("Data: Chicago Data Portal")

# ══════════════════════════════════════════════════════════════════════════════
# LOAD
# ══════════════════════════════════════════════════════════════════════════════
try:
    with st.spinner("Loading models & data..."):
        best_m1, best_m2, best_m3 = load_models()
        metas, scalers, label_map, train_summary, scorecard = load_meta()
        test_arrays = load_test_data()
    feat1 = metas["task1"]["feature_names"]
    feat2 = metas["task2"]["feature_names"]
    feat3 = metas["task3"]["feature_names"]
    class_names = [label_map[str(i)] for i in sorted(label_map.keys(), key=int)]
except Exception as e:
    st.error(f"❌ Failed to load models/data: {e}")
    st.info("Make sure `ml_data/` and `ml_models/` folders are in the app directory.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_predict, tab_dashboard = st.tabs(["🔮 Predict", "📊 Dashboard"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: PREDICT
# ─────────────────────────────────────────────────────────────────────────────
with tab_predict:
    st.header("🔮 Crime Predictions")
    st.caption("Fill in the crime details — all three models predict simultaneously.")

    with st.form("prediction_form"):
        st.subheader("📝 Crime Details")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**🕐 Time**")
            hour        = st.slider("Hour of Day", 0, 23, 12)
            month       = st.slider("Month", 1, 12, 6)
            day_of_week = st.selectbox("Day of Week", range(7),
                          format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])
            year        = st.slider("Year", 2010, 2024, 2019)
            season      = st.selectbox("Season", ["Spring", "Summer", "Fall", "Winter"])

        with col2:
            st.markdown("**📍 Location**")
            community_area = st.slider("Community Area (1-77)", 1, 77, 25)
            district       = st.slider("District", 1, 25, 8)
            beat           = st.number_input("Beat", 100, 2535, 835)
            ward           = st.slider("Ward", 1, 50, 27)
            lag1d          = st.number_input("Yesterday's crime count (area)", 0, 500, 15)
            lag3d          = st.number_input("3-day avg crime count (area)", 0, 500, 15)

        with col3:
            st.markdown("**🏷️ Crime Context**")
            is_night   = st.toggle("🌙 Night time crime", False)
            is_weekend = st.toggle("📅 Weekend", False)
            domestic   = st.toggle("🏠 Domestic incident", False)
            freq_type  = st.slider("Crime type frequency (0–1)", 0.0, 1.0, 0.15)
            freq_loc   = st.slider("Location frequency (0–1)", 0.0, 1.0, 0.10)
            te_type    = st.slider("Crime type arrest rate (0–1)", 0.0, 1.0, 0.25)

        submitted = st.form_submit_button("🚀 Run All Predictions",
                                          use_container_width=True, type="primary")

    if submitted:
        input_vals = {
            "Hour": hour, "Month": month, "DayOfWeek": day_of_week,
            "Year": year, "Season": season,
            "Community Area": community_area, "District": district,
            "Beat": beat, "Ward": ward,
            "IsNight": int(is_night), "IsWeekend": int(is_weekend), "Domestic": int(domestic),
            "freq_Primary Type": freq_type,
            "freq_Location Description": freq_loc,
            "te_Primary Type": te_type,
            "Lag1d": lag1d, "Lag3d": lag3d,
            "YearTrend": (year - 2001) / max(2024 - 2001, 1),
            "AreaMeanCount": lag1d,
        }

        r1, r2, r3 = st.columns(3)

        # Task 1
        with r1:
            st.markdown("### 🔵 Arrest Prediction")
            try:
                X1 = build_input_vector(feat1, input_vals)
                if scalers["task1"]: X1 = scalers["task1"].transform(X1)
                pred1  = best_m1.predict(X1)[0]
                prob1  = best_m1.predict_proba(X1)[0]
                arrest_prob = prob1[1]
                verdict = "🟢 Arrest Likely" if pred1 == 1 else "🔴 No Arrest"
                st.metric("Prediction", verdict)
                st.metric("Arrest Probability", f"{arrest_prob:.1%}")
                fig, ax = plt.subplots(figsize=(4, 1.4))
                ax.barh(["No Arrest", "Arrest"], [prob1[0], prob1[1]],
                        color=["#e07b6a", "#6ab4e0"], edgecolor="white", height=0.5)
                ax.set_xlim(0, 1)
                ax.axvline(0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
                ax.set_xlabel("Probability")
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()
            except Exception as e:
                st.error(f"Task 1 error: {e}")

        # Task 2
        with r2:
            st.markdown("### 🟢 Crime Type")
            try:
                X2 = build_input_vector(feat2, input_vals)
                if scalers["task2"]: X2 = scalers["task2"].transform(X2)
                pred2 = best_m2.predict(X2)[0]
                prob2 = best_m2.predict_proba(X2)[0]
                predicted_crime = class_names[int(pred2)]
                confidence = prob2.max()
                st.metric("Predicted Crime Type", predicted_crime)
                st.metric("Confidence", f"{confidence:.1%}")
                top5_idx = prob2.argsort()[-5:][::-1]
                fig, ax = plt.subplots(figsize=(4, 2.5))
                ax.barh([class_names[i] for i in top5_idx[::-1]],
                        prob2[top5_idx[::-1]],
                        color="#5ba85b", edgecolor="white", alpha=0.85)
                ax.set_xlabel("Probability")
                ax.set_title("Top 5 Types", fontsize=9)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()
            except Exception as e:
                st.error(f"Task 2 error: {e}")

        # Task 3
        with r3:
            st.markdown("### 🟣 Crime Count (Area)")
            try:
                X3 = build_input_vector(feat3, input_vals)
                if scalers["task3"]: X3 = scalers["task3"].transform(X3)
                raw_pred3 = best_m3.predict(X3)[0]
                pred3 = max(0, np.expm1(raw_pred3) if raw_pred3 < 20 else raw_pred3)
                level = "🔴 High" if pred3 > 30 else "🟡 Medium" if pred3 > 10 else "🟢 Low"
                st.metric("Predicted Crime Count", f"{pred3:.1f}")
                st.metric("Community Area", community_area)
                st.metric("Risk Level", level)
                fig, ax = plt.subplots(figsize=(4, 1.4))
                bar_color = "#e07b6a" if pred3 > 30 else "#f5a623" if pred3 > 10 else "#6ab4e0"
                ax.barh(["Predicted"], [pred3], color=bar_color, edgecolor="white", height=0.4)
                ax.axvline(15, color="black", linestyle="--", linewidth=0.8, alpha=0.5, label="Avg~15")
                ax.set_xlabel("Crime Count")
                ax.legend(fontsize=8)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()
            except Exception as e:
                st.error(f"Task 3 error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
with tab_dashboard:
    st.header("📊 Pipeline Dashboard")

    # KPI row
    st.subheader("🏆 Best Model Summary")
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    try:
        t1i = train_summary.get("task1", {})
        t2i = train_summary.get("task2", {})
        t3i = train_summary.get("task3", {})
        k1.metric("Task 1 Model",  t1i.get("best_model", "HGB"))
        k2.metric("Task 1 Score",  str(t1i.get("best_score", "—"))[:6])
        k3.metric("Task 2 Model",  t2i.get("best_model", "HGB"))
        k4.metric("Task 2 Score",  str(t2i.get("best_score", "—"))[:6])
        k5.metric("Task 3 Model",  t3i.get("best_model", "HGB"))
        k6.metric("Task 3 Score",  str(t3i.get("best_score", "—"))[:6])
    except Exception:
        st.info("Run all notebooks first to populate training summary.")

    st.divider()

    # Scorecard
    st.subheader("📋 Model Scorecard")
    try:
        st.dataframe(scorecard.style.format(precision=4), use_container_width=True)
    except Exception:
        st.dataframe(scorecard, use_container_width=True)

    st.divider()

    # Charts
    st.subheader("📈 Feature Importance")
    c1, c2 = st.columns(2)
    with c1:
        p = DATA_DIR / "06_tree_importance.png"
        if p.exists():
            st.image(str(p), caption="Feature Importance", use_container_width=True)
        else:
            st.info("Run notebook 06 to generate charts.")
    with c2:
        p = DATA_DIR / "06_permutation_importance.png"
        if p.exists():
            st.image(str(p), caption="Permutation Importance", use_container_width=True)
        else:
            st.info("Run notebook 06 to generate charts.")

    st.divider()

    # Live evaluation
    st.subheader("🧪 Live Test Set Evaluation")
    if st.button("▶️ Evaluate on test set", type="secondary"):
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, r2_score
        X_te1, y_te1 = test_arrays["task1"]["X_test"], test_arrays["task1"]["y_test"]
        X_te2, y_te2 = test_arrays["task2"]["X_test"], test_arrays["task2"]["y_test"]
        X_te3, y_te3 = test_arrays["task3"]["X_test"], test_arrays["task3"]["y_test"]
        with st.spinner("Evaluating..."):
            c1, c2, c3 = st.columns(3)
            try:
                y_p1 = best_m1.predict(X_te1)
                y_pb1 = best_m1.predict_proba(X_te1)[:, 1]
                c1.markdown("**🔵 Task 1 — Arrest**")
                c1.metric("AUC",      f"{roc_auc_score(y_te1, y_pb1):.4f}")
                c1.metric("F1",       f"{f1_score(y_te1, y_p1, average='weighted'):.4f}")
                c1.metric("Accuracy", f"{accuracy_score(y_te1, y_p1):.4f}")
            except Exception as e:
                c1.error(str(e))
            try:
                y_p2 = best_m2.predict(X_te2)
                c2.markdown("**🟢 Task 2 — Crime Type**")
                c2.metric("F1 (weighted)", f"{f1_score(y_te2, y_p2, average='weighted'):.4f}")
                c2.metric("Accuracy",      f"{accuracy_score(y_te2, y_p2):.4f}")
            except Exception as e:
                c2.error(str(e))
            try:
                y_p3 = np.expm1(best_m3.predict(X_te3))
                c3.markdown("**🟣 Task 3 — Crime Count**")
                c3.metric("RMSE", f"{np.sqrt(mean_squared_error(y_te3, y_p3)):.4f}")
                c3.metric("R²",   f"{r2_score(y_te3, y_p3):.4f}")
            except Exception as e:
                c3.error(str(e))

    st.divider()

    # Learning curves & error analysis
    st.subheader("📉 Learning Curves & Error Analysis")
    p = DATA_DIR / "06_learning_curves.png"
    if p.exists():
        st.image(str(p), caption="Bias-Variance Diagnosis", use_container_width=True)

    ea1, ea2 = st.columns(2)
    with ea1:
        p = DATA_DIR / "06_task1_error_analysis.png"
        if p.exists():
            st.image(str(p), caption="Task 1 Error Analysis", use_container_width=True)
    with ea2:
        p = DATA_DIR / "06_task3_error_analysis.png"
        if p.exists():
            st.image(str(p), caption="Task 3 Error Analysis", use_container_width=True)

    st.caption("🏙️ Chicago Crime ML Pipeline · Streamlit · Chicago Data Portal")
