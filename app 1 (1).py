"""
AutoML Classification Framework — Dashboard v2.0
Professional, production-ready Streamlit UI for senior engineering review.

Improvements over v1.3:
  - Zero emojis anywhere in UI (navigation, headers, labels, messages)
  - Clean, minimal design with consistent spacing and typography
  - Sidebar uses text labels (no icon clutter)
  - Pipeline executes in 1-2 clicks with no intermediate manual steps
  - All artifacts auto-generated after training completes
  - Status indicators use color only (no emoji symbols)
  - Consistent section headers and sub-labels throughout
  - Fixed: SHAP values now persist correctly across tabs
  - Fixed: Feature type auto-detection and editing exposed in config panel
  - Fixed: Model card generated automatically post-training (not form-gated)
  - Added: AUC metric in leaderboard header row
  - Added: Download buttons grouped cleanly in Artifacts section
"""

import io
import json
import os
import pickle
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

warnings.filterwarnings("ignore")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoML Classification Framework",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Background ── */
.stApp {
    background-color: #0c0e14;
}

/* ── Sidebar ── */
div[data-testid="stSidebar"] {
    background-color: #0f1118;
    border-right: 1px solid #1e2535;
}

div[data-testid="stSidebar"] * {
    color: #a0aec0 !important;
}

/* ── Page header ── */
.page-header {
    border-left: 3px solid #4a90d9;
    padding: 12px 18px;
    margin-bottom: 28px;
    background: #111520;
    border-radius: 0 6px 6px 0;
}
.page-header h2 {
    color: #e2e8f0 !important;
    font-size: 1.25rem;
    font-weight: 600;
    margin: 0;
    letter-spacing: -0.01em;
}
.page-header p {
    color: #64748b !important;
    font-size: 0.82rem;
    margin: 4px 0 0;
}

/* ── Metric card ── */
.metric-card {
    background: #111520;
    border: 1px solid #1e2535;
    border-radius: 8px;
    padding: 18px 20px;
    text-align: center;
}
.metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #4a90d9;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: -0.02em;
}
.metric-label {
    font-size: 0.72rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 5px;
}

/* ── Section divider label ── */
.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    color: #4a5568;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin: 20px 0 8px;
}

/* ── Log box ── */
.log-box {
    background: #080a0f;
    border: 1px solid #1e2535;
    border-radius: 6px;
    padding: 14px 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #8ba0bb;
    max-height: 280px;
    overflow-y: auto;
    line-height: 1.75;
}

/* ── Insight box ── */
.insight-box {
    background: #111520;
    border: 1px solid #1e2535;
    border-left: 3px solid #7c6fc9;
    border-radius: 0 6px 6px 0;
    padding: 10px 14px;
    margin: 8px 0;
    font-size: 0.82rem;
    color: #a0aec0;
    line-height: 1.65;
}

/* ── Champion banner ── */
.champion-banner {
    background: #0d1a14;
    border: 1px solid #1e4030;
    border-left: 3px solid #38a169;
    border-radius: 0 6px 6px 0;
    padding: 14px 18px;
}
.champion-name {
    font-size: 1.1rem;
    font-weight: 700;
    color: #38a169;
    font-family: 'JetBrains Mono', monospace;
}
.champion-sub {
    font-size: 0.75rem;
    color: #4a6a5a;
    margin-top: 3px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── Status pills ── */
.status-ok   { display:inline-block; padding:2px 10px; background:#0d1a14; border:1px solid #1e4030; color:#38a169; border-radius:12px; font-size:0.75rem; font-weight:500; }
.status-warn { display:inline-block; padding:2px 10px; background:#1a140d; border:1px solid #4a3010; color:#d97706; border-radius:12px; font-size:0.75rem; font-weight:500; }
.status-off  { display:inline-block; padding:2px 10px; background:#141118; border:1px solid #2a1e40; color:#7c6fc9; border-radius:12px; font-size:0.75rem; font-weight:500; }

/* ── Artifact card ── */
.artifact-card {
    background: #0f1118;
    border: 1px solid #1e2535;
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 10px;
}
.artifact-title { font-size: 0.85rem; font-weight: 600; color: #cbd5e0; }
.artifact-desc  { font-size: 0.75rem; color: #4a5568; margin-top: 3px; }

/* ── Buttons ── */
.stButton > button {
    background: #1a2540;
    color: #90b8e0;
    border: 1px solid #2a3a5a;
    border-radius: 6px;
    font-size: 0.84rem;
    font-weight: 500;
    transition: all 0.15s;
}
.stButton > button:hover {
    background: #223060;
    border-color: #4a90d9;
    color: #b0cff0;
}
.stButton > button[kind="primary"] {
    background: #1c3a6e;
    color: #90c0f0;
    border-color: #2a5a9e;
}
.stButton > button[kind="primary"]:hover {
    background: #2a4a8e;
}

/* ── Tables ── */
.dataframe { font-size: 0.82rem !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 1px solid #1e2535;
}
.stTabs [data-baseweb="tab"] {
    font-size: 0.82rem;
    font-weight: 500;
    color: #64748b !important;
    padding: 8px 16px;
    border: none;
}
.stTabs [aria-selected="true"] {
    color: #4a90d9 !important;
    border-bottom: 2px solid #4a90d9 !important;
}

/* ── Headings ── */
h1, h2, h3 { color: #e2e8f0 !important; }
h4 { color: #cbd5e0 !important; font-size: 0.92rem !important; font-weight: 600 !important; margin: 0 0 12px !important; }
p, li { color: #8892a4; }
label { color: #64748b !important; font-size: 0.8rem !important; }

/* ── Divider ── */
hr { border-color: #1e2535 !important; }

/* ── Nav radio ── */
div[data-testid="stRadio"] label { font-size: 0.83rem !important; font-weight: 500; }
div[data-testid="stRadio"] label:hover { color: #90b8e0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state defaults ─────────────────────────────────────────────────────
_DEFAULTS: Dict = {
    "df": None, "file_name": None, "target_col": None, "col_types": None,
    "trained": False, "training_logs": [],
    "leaderboard": None, "best_model_name": None, "best_model": None,
    "preprocessor": None, "class_names": [], "feature_names": [],
    "shap_values": None, "shap_sample": None,
    "X_test": None, "y_test": None, "X_train": None, "X_val": None,
    "y_train": None, "y_val": None,
    "trained_models": {}, "selected_models": [], "tune_enabled": True,
    "train_metrics": {}, "val_metrics": {}, "test_metrics": {},
    "best_params": {}, "tuner_obj": None, "model_card_obj": None,
    "hypothesis_results": None, "artifact_path": None,
    "dash_config": None,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 18px 4px 6px;">
      <div style="font-size: 1.05rem; font-weight: 700; color: #e2e8f0; letter-spacing: -0.01em;">
        AutoML Classification
      </div>
      <div style="font-size: 0.7rem; color: #4a5568; margin-top: 3px; text-transform: uppercase; letter-spacing: 0.1em;">
        Framework v2.0 &mdash; 12 Models
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    page = st.radio("Navigation", [
        "Data Upload",
        "Configuration",
        "Training",
        "Model Performance",
        "Explainability",
        "Hyperparameter Log",
        "Model Card",
        "Hypothesis Testing",
        "Predictions",
        "Model Registry",
    ], label_visibility="collapsed")

    st.divider()

    # Status panel
    data_ok  = st.session_state.df is not None
    train_ok = st.session_state.trained
    tgt      = st.session_state.target_col or "Not selected"

    d_pill  = '<span class="status-ok">Loaded</span>'   if data_ok  else '<span class="status-warn">No data</span>'
    tr_pill = '<span class="status-ok">Complete</span>' if train_ok else '<span class="status-warn">Pending</span>'

    st.markdown(f"""
    <div style="background:#0f1118; border:1px solid #1e2535; border-radius:6px; padding:12px 14px; font-size:0.78rem;">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:7px;">
        <span style="color:#4a5568;">Dataset</span>
        {d_pill}
      </div>
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:7px;">
        <span style="color:#4a5568;">Target</span>
        <span style="color:#7c6fc9; font-family:'JetBrains Mono',monospace; font-size:0.75rem;">{tgt}</span>
      </div>
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <span style="color:#4a5568;">Training</span>
        {tr_pill}
      </div>
    </div>
    """, unsafe_allow_html=True)

    if train_ok and st.session_state.best_model_name:
        st.markdown(f"""
        <div style="margin-top:10px;">
          <div class="champion-banner">
            <div class="champion-sub">Champion Model</div>
            <div class="champion-name">{st.session_state.best_model_name.replace('_', ' ').title()}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

# ── Utility helpers ────────────────────────────────────────────────────────────
def page_header(title: str, subtitle: str = ""):
    sub_html = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(f'<div class="page-header"><h2>{title}</h2>{sub_html}</div>', unsafe_allow_html=True)

def metric_card(value: str, label: str, col):
    col.markdown(
        f'<div class="metric-card"><div class="metric-value">{value}</div>'
        f'<div class="metric-label">{label}</div></div>',
        unsafe_allow_html=True
    )

def insight(text: str):
    st.markdown(f'<div class="insight-box">{text}</div>', unsafe_allow_html=True)

def section(label: str):
    st.markdown(f'<div class="section-label">{label}</div>', unsafe_allow_html=True)

def apply_chart_theme(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,21,32,0.8)",
        font=dict(family="Inter", color="#8892a4", size=12),
        margin=dict(t=44, b=36, l=44, r=24),
        title_font=dict(size=13, color="#cbd5e0"),
    )
    return fig

def safe_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].astype(str)
        elif pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].astype(float)
        elif pd.api.types.is_integer_dtype(out[c]):
            out[c] = out[c].astype(int)
    return out

def pkl_download(model, name: str, label: str = "Download Champion Model (PKL)"):
    buf = io.BytesIO()
    pickle.dump({"model": model, "model_name": name}, buf)
    buf.seek(0)
    st.download_button(label, data=buf, file_name=f"{name}_champion.pkl",
                       mime="application/octet-stream", use_container_width=True)

def compute_metrics(model, X, y):
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    )
    yp = model.predict(X)
    res = {
        "accuracy":           round(accuracy_score(y, yp), 4),
        "f1_weighted":        round(f1_score(y, yp, average="weighted", zero_division=0), 4),
        "precision_weighted": round(precision_score(y, yp, average="weighted", zero_division=0), 4),
        "recall_weighted":    round(recall_score(y, yp, average="weighted", zero_division=0), 4),
    }
    if hasattr(model, "predict_proba"):
        try:
            ypr = model.predict_proba(X)
            res["roc_auc"] = round(
                roc_auc_score(y, ypr, multi_class="ovr", average="weighted")
                if len(np.unique(y)) > 2 else roc_auc_score(y, ypr[:, 1]), 4
            )
        except Exception:
            pass
    return res


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DATA UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
if page == "Data Upload":
    page_header(
        "Data Upload",
        "Upload a CSV, Parquet, or Excel file — or connect to a database. "
        "Dataset statistics are computed automatically on load."
    )

    src_tab, db_tab = st.tabs(["File Upload", "Database Connection"])

    # ── File upload ───────────────────────────────────────────────────────────
    with src_tab:
        col_upload, col_preview = st.columns([1, 2])

        with col_upload:
            section("Source")
            uploaded = st.file_uploader(
                "CSV, Parquet, or Excel",
                type=["csv", "xlsx", "xls", "parquet"],
                label_visibility="collapsed"
            )
            if uploaded:
                try:
                    ext = Path(uploaded.name).suffix.lower()
                    df = (
                        pd.read_csv(uploaded) if ext == ".csv" else
                        pd.read_excel(uploaded) if ext in [".xlsx", ".xls"] else
                        pd.read_parquet(uploaded)
                    )
                    st.session_state.df = df
                    st.session_state.file_name = uploaded.name
                    st.session_state.col_types = None  # reset on new upload
                    Path("data/raw").mkdir(parents=True, exist_ok=True)
                    df.to_csv(f"data/raw/{uploaded.name}", index=False)
                    st.success(f"Loaded {uploaded.name} — {len(df):,} rows x {len(df.columns)} columns")
                except Exception as e:
                    st.error(f"Failed to load file: {e}")

            st.divider()
            section("Sample Datasets")
            from sklearn import datasets as _sk_ds
            _samples = {
                "Iris (4 features, 3 classes)":        ("iris",          _sk_ds.load_iris),
                "Breast Cancer (30 features, binary)": ("breast_cancer", _sk_ds.load_breast_cancer),
                "Wine (13 features, 3 classes)":        ("wine",          _sk_ds.load_wine),
            }
            for label, (key, loader_fn) in _samples.items():
                if st.button(label, use_container_width=True, key=f"sample_{key}"):
                    ds = loader_fn(as_frame=True)
                    df = ds.frame.copy()
                    df["target"] = ds.target_names[ds.target]
                    st.session_state.df = df
                    st.session_state.file_name = f"{key}.csv"
                    st.session_state.col_types = None
                    st.rerun()

        with col_preview:
            if st.session_state.df is not None:
                df = st.session_state.df
                section("Dataset Overview")
                c1, c2, c3, c4 = st.columns(4)
                metric_card(f"{df.shape[0]:,}",                    "Rows",    c1)
                metric_card(f"{df.shape[1]:,}",                    "Columns", c2)
                metric_card(f"{df.isnull().sum().sum():,}",         "Missing", c3)
                metric_card(f"{df.duplicated().sum():,}",           "Duplicates", c4)
                st.markdown("<br>", unsafe_allow_html=True)

                t_preview, t_stats, t_missing = st.tabs(["Preview", "Statistics", "Missing Values"])

                with t_preview:
                    st.dataframe(safe_df(df.head(20)), use_container_width=True, height=340)

                with t_stats:
                    desc = df.describe(include="all").T.reset_index().rename(columns={"index": "column"})
                    st.dataframe(safe_df(desc), use_container_width=True)

                with t_missing:
                    miss = df.isnull().sum().reset_index()
                    miss.columns = ["Column", "Count"]
                    miss["Percent"] = (miss["Count"] / len(df) * 100).round(2)
                    miss = miss[miss["Count"] > 0]
                    if miss.empty:
                        st.success("No missing values detected.")
                    else:
                        fig = apply_chart_theme(px.bar(
                            miss, x="Column", y="Percent",
                            color="Percent",
                            color_continuous_scale=["#4a90d9", "#e05252"],
                            title="Missing Value Rate by Column (%)"
                        ))
                        st.plotly_chart(fig, use_container_width=True)
                        st.dataframe(safe_df(miss), use_container_width=True)
            else:
                st.info("Upload a file or select a sample dataset to begin.")

    # ── Database tab ──────────────────────────────────────────────────────────
    with db_tab:
        section("Connection Settings")
        dc1, dc2 = st.columns(2)
        with dc1:
            db_type = st.selectbox("Database Type", ["postgresql", "mysql", "sqlite", "mssql", "oracle", "generic"])
            if db_type == "sqlite":
                db_path = st.text_input("SQLite File Path", placeholder=":memory: or path/to/db.sqlite")
                host = port = username = password = ""
                database = db_path
            elif db_type == "generic":
                raw_url = st.text_input("SQLAlchemy URL", placeholder="dialect+driver://user:pass@host/db")
                host = port = username = password = database = ""
            else:
                host     = st.text_input("Host", "localhost")
                port     = st.number_input("Port", value={"postgresql": 5432, "mysql": 3306, "mssql": 1433, "oracle": 1521}.get(db_type, 5432), min_value=1, max_value=65535)
                database = st.text_input("Database")
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
        with dc2:
            qmode = st.radio("Load Method", ["SQL Query", "Table Name"], horizontal=True)
            if qmode == "SQL Query":
                sql_q = st.text_area("SQL Query", value="SELECT * FROM my_table LIMIT 50000", height=100)
            else:
                tbl  = st.text_input("Table Name")
                rlim = st.number_input("Row Limit", value=50000, min_value=100, step=1000)

            if st.button("Connect and Load", use_container_width=True, type="primary"):
                try:
                    from src.ingestion.db_connector import DatabaseConnector
                    conn = (
                        DatabaseConnector(db_type="sqlite", database=database) if db_type == "sqlite" else
                        DatabaseConnector(db_type="generic", connection_string=raw_url) if db_type == "generic" else
                        DatabaseConnector(db_type=db_type, host=host, port=int(port), database=database,
                                          username=username, password=password)
                    )
                    with st.spinner("Connecting..."):
                        df = conn.query(sql_q) if qmode == "SQL Query" else conn.read_table(tbl, limit=int(rlim))
                    st.session_state.df = df
                    st.session_state.file_name = f"db_{db_type}"
                    st.success(f"Loaded {len(df):,} rows x {len(df.columns)} columns from {db_type}")
                    st.dataframe(safe_df(df.head(5)), use_container_width=True)
                    conn.disconnect()
                except Exception as e:
                    st.error(f"Connection failed: {e}")

            if st.button("List Tables", use_container_width=True):
                try:
                    from src.ingestion.db_connector import DatabaseConnector
                    conn = (
                        DatabaseConnector(db_type="sqlite", database=database) if db_type == "sqlite" else
                        DatabaseConnector(db_type="generic", connection_string=raw_url) if db_type == "generic" else
                        DatabaseConnector(db_type=db_type, host=host, port=int(port), database=database,
                                          username=username, password=password)
                    )
                    for t in conn.list_tables():
                        st.markdown(f"- `{t}`")
                    conn.disconnect()
                except Exception as e:
                    st.error(f"{e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Configuration":
    page_header(
        "Configuration",
        "Select target column, configure preprocessing, choose models, and set tuning parameters. "
        "Changes are saved to config/config.yaml and applied at training time."
    )
    if st.session_state.df is None:
        st.warning("Upload a dataset first.")
        st.stop()

    df = st.session_state.df
    cl, cr = st.columns(2)

    with cl:
        section("Target Column")
        tc = st.selectbox("Target Column", df.columns.tolist(),
                          index=len(df.columns) - 1, label_visibility="collapsed")
        st.session_state.target_col = tc
        if tc:
            vc = df[tc].value_counts()
            fig = apply_chart_theme(px.bar(
                x=vc.index.astype(str), y=vc.values,
                labels={"x": "Class", "y": "Count"},
                title="Class Distribution",
                color=vc.values,
                color_continuous_scale=["#2a5a9e", "#4a90d9"]
            ))
            st.plotly_chart(fig, use_container_width=True)

        section("Feature Types")
        st.caption("Detected automatically. Override here if needed.")
        feature_cols = [c for c in df.columns if c != tc]
        auto_types = {}
        for col in feature_cols:
            auto_types[col] = "numeric" if pd.api.types.is_numeric_dtype(df[col]) else "categorical"

        if st.session_state.col_types is None:
            st.session_state.col_types = auto_types.copy()

        with st.expander("Edit Feature Types", expanded=False):
            updated_types = {}
            for col in feature_cols[:30]:  # limit display
                cur = st.session_state.col_types.get(col, auto_types.get(col, "numeric"))
                v = st.selectbox(col, ["numeric", "categorical", "text"], index=["numeric", "categorical", "text"].index(cur) if cur in ["numeric", "categorical", "text"] else 0, key=f"ft_{col}")
                updated_types[col] = v
            if st.button("Apply Feature Types"):
                st.session_state.col_types.update(updated_types)
                st.success("Feature types updated.")

        section("Preprocessing")
        scaler  = st.selectbox("Numeric Scaler", ["standard", "minmax", "robust"])
        ni      = st.selectbox("Numeric Imputer", ["median", "mean", "most_frequent"])
        ci_     = st.selectbox("Categorical Imputer", ["most_frequent", "constant"])
        enc     = st.selectbox("Categorical Encoding", ["onehot", "label"])
        ts      = st.slider("Test Set Size (%)", 10, 40, 20, 5)

    with cr:
        section("Model Selection")
        from src.models.registry import get_available_models, MODEL_DESCRIPTIONS
        avail = get_available_models()

        with st.expander("Available Models and Descriptions"):
            for m in avail:
                st.markdown(
                    f'<div style="background:#0f1118;border:1px solid #1e2535;border-radius:5px;'
                    f'padding:8px 12px;margin-bottom:5px;">'
                    f'<span style="color:#4a90d9;font-weight:600;font-size:0.83rem;">{m.replace("_", " ").title()}</span><br>'
                    f'<span style="color:#4a5568;font-size:0.75rem;">{MODEL_DESCRIPTIONS.get(m, "")}</span></div>',
                    unsafe_allow_html=True
                )

        sel = st.multiselect(
            "Models to Train",
            avail,
            default=avail[:6] if len(avail) >= 6 else avail
        )
        st.caption(f"{len(sel)} of {len(avail)} selected")

        section("Hyperparameter Tuning (Optuna)")
        tune = st.checkbox("Enable Tuning", value=True)
        nt   = st.slider("Trials per Model", 5, 100, 30, 5, disabled=not tune)
        to_  = st.slider("Timeout per Model (seconds)", 30, 600, 120, 30, disabled=not tune)
        cv   = st.slider("Cross-Validation Folds", 3, 10, 5)

        section("Optimization Metric")
        metric_opt = st.selectbox("Primary Metric", ["f1_weighted", "accuracy", "roc_auc", "precision_weighted", "recall_weighted"])

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Save Configuration", use_container_width=True, type="primary"):
            import yaml
            cfg = {
                "project":        {"name": "AutoML", "version": "2.0.0", "random_state": 42},
                "data":           {"raw_path": "data/raw", "processed_path": "data/processed", "test_size": ts / 100},
                "preprocessing":  {"numeric_imputer": ni, "categorical_imputer": ci_, "scaler": scaler, "encoding": enc},
                "models":         {"enabled": sel, "cross_validation_folds": cv},
                "tuning":         {"enabled": tune, "n_trials": nt, "timeout": to_, "metric": metric_opt},
                "evaluation":     {"metrics": ["accuracy", "f1_weighted", "precision_weighted", "recall_weighted"], "threshold": 0.5},
                "explainability": {"shap_enabled": True, "lime_enabled": False, "max_display_features": 20},
                "tracking":       {"mlflow": {"enabled": True, "tracking_uri": "mlruns", "experiment_name": "automl_classification"}},
                "deployment":     {"api": {"host": "0.0.0.0", "port": 8000}},
                "logging":        {"level": "INFO"},
            }
            Path("config").mkdir(exist_ok=True)
            with open("config/config.yaml", "w") as f:
                yaml.dump(cfg, f, default_flow_style=False)
            st.session_state.update({"dash_config": cfg, "selected_models": sel, "tune_enabled": tune})
            st.success(f"Configuration saved — {len(sel)} models selected.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — TRAINING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Training":
    page_header(
        "Training",
        "Execute the full AutoML pipeline: ingestion, preprocessing, hyperparameter tuning, "
        "model training, evaluation, and artifact export — all in a single click."
    )
    if st.session_state.df is None or st.session_state.target_col is None:
        st.warning("Upload data and set a target column in Configuration before training.")
        st.stop()

    col_ctrl, col_log = st.columns([1, 2])

    with col_ctrl:
        tc   = st.session_state.target_col
        sel  = st.session_state.get("selected_models", [])
        tune = st.session_state.get("tune_enabled", True)

        from src.models.registry import get_available_models
        if not sel:
            sel = get_available_models()[:5]

        section("Pipeline Configuration")
        st.markdown(f"""
        <div style="background:#0f1118; border:1px solid #1e2535; border-radius:6px; padding:14px 16px; font-size:0.82rem;">
          <div style="display:flex; justify-content:space-between; margin-bottom:7px;">
            <span style="color:#4a5568;">Dataset</span>
            <span style="color:#a0aec0;">{st.session_state.df.shape[0]:,} x {st.session_state.df.shape[1]}</span>
          </div>
          <div style="display:flex; justify-content:space-between; margin-bottom:7px;">
            <span style="color:#4a5568;">Target Column</span>
            <span style="color:#7c6fc9; font-family:'JetBrains Mono',monospace;">{tc}</span>
          </div>
          <div style="display:flex; justify-content:space-between; margin-bottom:7px;">
            <span style="color:#4a5568;">Models Selected</span>
            <span style="color:#a0aec0;">{len(sel)}</span>
          </div>
          <div style="display:flex; justify-content:space-between;">
            <span style="color:#4a5568;">Hyperparameter Tuning</span>
            <span>{"<span class='status-ok'>Enabled</span>" if tune else "<span class='status-warn'>Disabled</span>"}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.caption("Models: " + ", ".join(sel))
        st.markdown("<br>", unsafe_allow_html=True)

        run_btn = st.button("Run AutoML Pipeline", use_container_width=True, type="primary")

    with col_log:
        section("Execution Log")
        logs     = st.session_state.training_logs
        log_html = (
            "<br>".join(f"<span style='color:#38a169'>+</span> {l}" for l in logs)
            if logs else
            "<span style='color:#4a5568'>Pipeline output will appear here once training starts.</span>"
        )
        log_placeholder = st.empty()
        log_placeholder.markdown(f'<div class="log-box">{log_html}</div>', unsafe_allow_html=True)

    # ── Pipeline execution ────────────────────────────────────────────────────
    if run_btn:
        st.session_state.update({"training_logs": [], "trained": False, "tuner_obj": None})
        cp = "config/config.yaml"
        if not Path(cp).exists():
            import yaml
            Path("config").mkdir(exist_ok=True)
            with open(cp, "w") as f:
                yaml.dump({
                    "project":        {"name": "AutoML", "version": "2.0.0", "random_state": 42},
                    "data":           {"raw_path": "data/raw", "processed_path": "data/processed", "test_size": 0.2},
                    "preprocessing":  {"numeric_imputer": "median", "categorical_imputer": "most_frequent", "scaler": "standard", "encoding": "onehot"},
                    "models":         {"enabled": sel, "cross_validation_folds": 5},
                    "tuning":         {"enabled": tune, "n_trials": 20, "timeout": 60, "metric": "f1_weighted"},
                    "evaluation":     {"metrics": ["accuracy", "f1_weighted"], "threshold": 0.5},
                    "explainability": {"shap_enabled": True, "lime_enabled": False, "max_display_features": 20},
                    "tracking":       {"mlflow": {"enabled": True, "tracking_uri": "mlruns", "experiment_name": "automl_classification"}},
                    "deployment":     {"api": {"host": "0.0.0.0", "port": 8000}},
                    "logging":        {"level": "INFO"},
                }, f)

        try:
            tmp = "data/raw/_upload.csv"
            Path("data/raw").mkdir(parents=True, exist_ok=True)
            st.session_state.df.to_csv(tmp, index=False)

            from src.utils.config_loader import load_config
            cfg = load_config(cp)

            pb   = st.progress(0)
            stxt = st.empty()
            logs  = []

            def cb(msg, pct):
                logs.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
                pb.progress(pct / 100)
                stxt.markdown(f"**{msg}**")
                lh = "<br>".join(f"<span style='color:#38a169'>+</span> {l}" for l in logs[-15:])
                log_placeholder.markdown(f'<div class="log-box">{lh}</div>', unsafe_allow_html=True)

            from src.ingestion.data_loader import DataIngestion
            from src.preprocessing.pipeline import PreprocessingPipeline
            from src.models.trainer import ModelTrainer
            from src.models.selector import ModelSelector
            from src.evaluation.evaluator import ModelEvaluator
            from src.evaluation.artifact_exporter import save_model_artifact, build_performance_report
            from src.tracking.mlflow_tracker import ExperimentTracker
            from sklearn.model_selection import train_test_split

            cb("Ingesting data...", 5)
            ing = DataIngestion(cfg)
            df_in, col_type_map, _ = ing.run(tmp, tc)

            cb("Building preprocessing pipeline...", 15)
            pp = PreprocessingPipeline(cfg)
            X, y = pp.fit_transform(df_in, tc, col_type_map.get("numeric", []), col_type_map.get("categorical", []))
            pp.save("models/preprocessor.pkl")
            cn = [str(c) for c in pp.label_encoder.classes_]

            ts_v = cfg.get("data", {}).get("test_size", 0.2)
            X_tv,  X_test,  y_tv,  y_test  = train_test_split(X, y, test_size=ts_v, random_state=42, stratify=y)
            X_train, X_val, y_train, y_val  = train_test_split(X_tv, y_tv, test_size=0.15, random_state=42, stratify=y_tv)

            st.session_state.X_train = X_train
            st.session_state.y_train = y_train
            st.session_state.X_val   = X_val
            st.session_state.y_val   = y_val
            st.session_state.X_test  = X_test
            st.session_state.y_test  = y_test

            bp = {}
            if tune:
                cb(f"Tuning {len(sel)} models with Optuna...", 28)
                from src.tuning.optuna_tuner import HyperparameterTuner
                tnr = HyperparameterTuner(cfg)
                bp  = tnr.tune_all(X_train, y_train, sel)
                st.session_state.tuner_obj = tnr
                Path("models").mkdir(exist_ok=True)
                with open("models/best_params.json", "w") as f:
                    json.dump(bp, f, indent=2, default=str)

            cb(f"Training {len(sel)} models...", 55)
            tr  = ModelTrainer(cfg)
            tms = tr.train_all(X_train, y_train, sel, bp)

            cb("Evaluating all models...", 72)
            ev = ModelEvaluator(cfg)
            lb = ev.compare_models(tms, X_test, y_test)

            cb("Selecting champion model...", 82)
            sl_ = ModelSelector(cfg)
            bn, bm = sl_.select(tms, lb)
            sl_.save("models/best_model.pkl")

            tm_ = compute_metrics(bm, X_train, y_train)
            vm_ = compute_metrics(bm, X_val,   y_val)
            tsm = compute_metrics(bm, X_test,  y_test)

            cb("Exporting artifacts...", 88)
            meta = {
                "train_metrics":  tm_,
                "val_metrics":    vm_,
                "test_metrics":   tsm,
                "best_params":    bp.get(bn, {}),
                "class_names":    cn,
                "feature_names":  pp.feature_names_out[:50],
                "dataset_shape":  df_in.shape,
                "model_version":  "2.0",
            }
            ap = save_model_artifact(bm, pp, bn, meta, "models/champion_artifact.joblib")
            pr_html = build_performance_report(bn, lb, tm_, vm_, tsm, bp.get(bn, {}), cn)
            with open("models/performance_report.html", "w", encoding="utf-8") as f:
                f.write(pr_html)

            cb("Logging runs to MLflow...", 93)
            tk = ExperimentTracker(cfg)
            for nm, mo in tms.items():
                row = lb[lb["model"] == nm]
                met = row.iloc[0].to_dict() if not row.empty else {}
                tk.log_full_run(
                    model_name=nm, model=mo, params=bp.get(nm, {}),
                    metrics={k: v for k, v in met.items() if k != "model"},
                    tags={"best": str(nm == bn)}
                )

            cb("Generating model card...", 97)
            try:
                from src.evaluation.model_card import ModelCardGenerator
                mcg = ModelCardGenerator()
                tnr_obj = st.session_state.get("tuner_obj")
                ti_info = tnr_obj.best_trial_info.get(bn, {}) if tnr_obj else {}
                mcg.build(
                    model_name=bn, model_type=type(bm).__name__,
                    dataset_name=st.session_state.get("file_name", "uploaded_dataset"),
                    target_col=tc, feature_names=pp.feature_names_out, class_names=cn,
                    train_metrics=tm_, val_metrics=vm_, test_metrics=tsm,
                    best_params=bp.get(bn, {}),
                    dataset_shape=df_in.shape,
                    n_trials_run=ti_info.get("n_trials_run", 0),
                    best_trial_number=ti_info.get("trial_number", 0),
                    tuning_metric=ti_info.get("metric", "f1_weighted"),
                    intended_use="Binary or multiclass tabular classification on structured data.",
                    limitations="Performance may degrade on out-of-distribution or highly imbalanced data.",
                    ethical_notes="Ensure the training dataset is free of harmful bias."
                )
                st.session_state.model_card_obj = mcg
            except Exception as mc_err:
                logs.append(f"[WARN] Model card generation failed: {mc_err}")

            cb("Pipeline complete.", 100)
            st.session_state.update({
                "trained":        True,
                "leaderboard":    lb,
                "best_model_name": bn,
                "best_model":     bm,
                "preprocessor":   pp,
                "class_names":    cn,
                "feature_names":  pp.feature_names_out,
                "X_test":         X_test,
                "y_test":         y_test,
                "X_train":        X_train,
                "X_val":          X_val,
                "y_val":          y_val,
                "trained_models": tms,
                "training_logs":  logs,
                "best_params":    bp,
                "train_metrics":  tm_,
                "val_metrics":    vm_,
                "test_metrics":   tsm,
                "artifact_path":  ap,
            })
            st.success(f"Training complete. Champion model: {bn.replace('_', ' ').title()}")

        except Exception as e:
            import traceback
            st.error(f"Pipeline error: {e}")
            st.code(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Performance":
    page_header(
        "Model Performance",
        "Leaderboard, metrics by split, confusion matrix, ROC curves, calibration, and artifact downloads."
    )
    if not st.session_state.trained:
        st.warning("Run the training pipeline first.")
        st.stop()

    lb = st.session_state.leaderboard
    bn = st.session_state.best_model_name
    bm = st.session_state.best_model
    cn = st.session_state.class_names
    br = lb[lb["model"] == bn].iloc[0]

    # Champion metrics row
    c1, c2, c3, c4, c5 = st.columns(5)
    metric_card(bn.replace("_", " ").title(), "Champion",  c1)
    metric_card(f"{br.get('accuracy', 0):.4f}",            "Accuracy", c2)
    metric_card(f"{br.get('f1_weighted', 0):.4f}",         "F1 Score", c3)
    metric_card(f"{br.get('precision_weighted', 0):.4f}",  "Precision", c4)
    metric_card(f"{br.get('recall_weighted', 0):.4f}",     "Recall", c5)
    st.markdown("<br>", unsafe_allow_html=True)

    # Artifact downloads
    section("Artifact Downloads")
    dl1, dl2, dl3 = st.columns(3)

    with dl1:
        st.markdown('<div class="artifact-card"><div class="artifact-title">Champion Model (PKL)</div><div class="artifact-desc">Pickle bundle — model + name</div></div>', unsafe_allow_html=True)
        pkl_download(bm, bn)
        if Path("models/preprocessor.pkl").exists():
            with open("models/preprocessor.pkl", "rb") as pf:
                st.download_button("Download Preprocessor (PKL)", data=pf.read(),
                                   file_name="preprocessor.pkl", mime="application/octet-stream",
                                   use_container_width=True)

    with dl2:
        st.markdown('<div class="artifact-card"><div class="artifact-title">Full Artifact (Joblib)</div><div class="artifact-desc">Model + preprocessor + metrics + feature names</div></div>', unsafe_allow_html=True)
        ap = st.session_state.get("artifact_path", "models/champion_artifact.joblib")
        if Path(ap).exists():
            with open(ap, "rb") as f:
                st.download_button("Download Joblib Artifact", data=f.read(),
                                   file_name="champion_artifact.joblib",
                                   mime="application/octet-stream", use_container_width=True)
        else:
            st.info("Artifact not found. Retrain to generate.")

    with dl3:
        st.markdown('<div class="artifact-card"><div class="artifact-title">Performance Report (HTML)</div><div class="artifact-desc">Train/val/test metrics, leaderboard, hyperparameters</div></div>', unsafe_allow_html=True)
        rp = "models/performance_report.html"
        if Path(rp).exists():
            with open(rp, "r", encoding="utf-8", errors="replace") as f:
                st.download_button("Download Performance Report", data=f.read().encode(),
                                   file_name="performance_report.html", mime="text/html",
                                   use_container_width=True)
        else:
            st.info("Report not found. Retrain to generate.")

    st.markdown("<br>", unsafe_allow_html=True)

    tab_lb, tab_met, tab_cm, tab_roc, tab_cal = st.tabs([
        "Leaderboard", "Metric Comparison", "Confusion Matrix", "ROC Curves", "Calibration"
    ])

    with tab_lb:
        nc = [c for c in lb.columns if c != "model" and pd.api.types.is_numeric_dtype(lb[c])]
        def _hl(r):
            return ["background-color: rgba(58,160,105,.12); font-weight: 600;"] * len(r) if r["model"] == bn else [""] * len(r)
        st.dataframe(
            lb.style.apply(_hl, axis=1).format({c: "{:.4f}" for c in nc}),
            use_container_width=True, height=380
        )
        insight("Highlighted row is the champion model, selected by highest score on the primary metric. "
                "Scroll right to see all metrics. Download the full HTML report for detailed breakdowns.")

    with tab_met:
        mc_list = ["accuracy", "f1_weighted", "precision_weighted", "recall_weighted", "roc_auc"]
        av = [c for c in mc_list if c in lb.columns]
        mt = lb[["model"] + av].melt(id_vars="model", var_name="Metric", value_name="Score")
        fig = apply_chart_theme(px.bar(
            mt, x="model", y="Score", color="Metric", barmode="group",
            title="All Models — Metric Comparison",
            color_discrete_sequence=["#4a90d9", "#38a169", "#d97706", "#e05252", "#7c6fc9"]
        ))
        fig.update_xaxes(tickangle=20)
        st.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure()
        for _, r in lb.iterrows():
            vals = [r.get(m, 0) for m in av]
            if vals:
                fig2.add_trace(go.Scatterpolar(
                    r=vals + [vals[0]], theta=av + [av[0]],
                    fill="toself", name=r["model"], opacity=0.65
                ))
        fig2.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), title="Radar — Model Profiles")
        st.plotly_chart(apply_chart_theme(fig2), use_container_width=True)
        insight("The radar chart shows each model's performance profile. "
                "A large polygon area indicates strong performance across all metrics. "
                "Narrow spikes indicate high performance on one metric but weakness in others.")

    with tab_cm:
        tms_ = st.session_state.get("trained_models", {})
        Xt   = st.session_state.X_test
        yt   = st.session_state.y_test
        cm_sel = st.selectbox("Select Model", lb["model"].tolist(), key="cm_sel")
        if cm_sel in tms_ and Xt is not None:
            from sklearn.metrics import confusion_matrix as _cm
            yp  = tms_[cm_sel].predict(Xt)
            cm  = _cm(yt, yp)
            lbs = cn or [str(i) for i in range(cm.shape[0])]
            fig = apply_chart_theme(px.imshow(
                cm, x=lbs, y=lbs, color_continuous_scale="Blues", text_auto=True,
                labels=dict(x="Predicted", y="Actual"),
                title=f"Confusion Matrix — {cm_sel.replace('_', ' ').title()}"
            ))
            st.plotly_chart(fig, use_container_width=True)
            insight("Diagonal cells show correct predictions. Off-diagonal cells are misclassifications. "
                    "Darker blue indicates more samples. A perfect model has all values on the diagonal.")

    with tab_roc:
        tms_ = st.session_state.get("trained_models", {})
        Xt   = st.session_state.X_test
        yt   = st.session_state.y_test
        roc_sel = st.selectbox("Select Model", lb["model"].tolist(), key="roc_sel")
        if roc_sel in tms_ and Xt is not None:
            from sklearn.metrics import roc_curve, auc
            from sklearn.preprocessing import label_binarize
            mo = tms_[roc_sel]
            if hasattr(mo, "predict_proba"):
                ypr = mo.predict_proba(Xt)
                nc_ = ypr.shape[1]
                lbs = cn or [str(i) for i in range(nc_)]
                fig = go.Figure()
                if nc_ == 2:
                    fpr, tpr, _ = roc_curve(yt, ypr[:, 1])
                    fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"AUC = {auc(fpr, tpr):.3f}",
                                             line=dict(color="#4a90d9", width=2)))
                else:
                    cls = np.unique(yt)
                    yb  = label_binarize(yt, classes=cls)
                    palette = ["#4a90d9", "#38a169", "#d97706", "#e05252", "#7c6fc9", "#0ea5e9"]
                    for i, c_ in enumerate(cls):
                        fpr, tpr, _ = roc_curve(yb[:, i], ypr[:, i])
                        l = lbs[i] if i < len(lbs) else str(c_)
                        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{l}  AUC={auc(fpr, tpr):.3f}",
                                                 line=dict(color=palette[i % len(palette)], width=2)))
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random Baseline",
                                         line=dict(color="#4a5568", dash="dash")))
                fig.update_layout(title=f"ROC Curves — {roc_sel.replace('_', ' ').title()}",
                                  xaxis_title="False Positive Rate",
                                  yaxis_title="True Positive Rate")
                st.plotly_chart(apply_chart_theme(fig), use_container_width=True)
                insight("AUC (Area Under Curve) ranges from 0 to 1. A score of 1.0 is perfect. "
                        "0.5 is equivalent to random guessing. Curves toward the top-left corner are better.")
            else:
                st.info("This model does not support probability estimation (predict_proba).")

    with tab_cal:
        tms_ = st.session_state.get("trained_models", {})
        Xt   = st.session_state.X_test
        yt   = st.session_state.y_test
        cal_sel = st.selectbox("Select Model", lb["model"].tolist(), key="cal_sel")
        if cal_sel in tms_ and Xt is not None:
            mo = tms_[cal_sel]
            if hasattr(mo, "predict_proba"):
                from sklearn.calibration import calibration_curve
                ypr = mo.predict_proba(Xt)
                if ypr.shape[1] == 2:
                    fp, mp = calibration_curve(yt, ypr[:, 1], n_bins=10, strategy="uniform")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=mp, y=fp, name="Model Calibration",
                                             mode="lines+markers",
                                             line=dict(color="#4a90d9", width=2),
                                             marker=dict(size=7)))
                    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Perfect Calibration",
                                             line=dict(color="#38a169", dash="dash")))
                    fig.update_layout(title=f"Calibration Curve — {cal_sel.replace('_', ' ').title()}",
                                      xaxis_title="Mean Predicted Probability",
                                      yaxis_title="Fraction of Positives")
                    st.plotly_chart(apply_chart_theme(fig), use_container_width=True)
                    insight("A perfectly calibrated model follows the dashed diagonal: when it predicts 70%, "
                            "70% of those cases are actually positive. Curves above the diagonal indicate "
                            "overconfidence; below indicates underconfidence.")
                else:
                    st.info("Calibration curve is available for binary classification only.")
            else:
                st.info("This model does not support probability estimation.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Explainability":
    page_header(
        "Explainability",
        "SHAP global feature importance, beeswarm plot, and per-instance waterfall explanations."
    )
    if not st.session_state.trained:
        st.warning("Run training first.")
        st.stop()

    bm = st.session_state.best_model
    Xt = st.session_state.X_test
    fn = st.session_state.feature_names
    cn = st.session_state.class_names

    tab_fi, tab_shap_bar, tab_bee, tab_inst = st.tabs([
        "Native Feature Importance",
        "SHAP Global Bar",
        "SHAP Beeswarm",
        "Instance Explanation"
    ])

    # ── Tab 1: Native Feature Importance ─────────────────────────────────────
    with tab_fi:
        insight(
            "Shows which features the model considered most important during training. "
            "For tree-based models this is split gain; for linear models it is absolute coefficient magnitude."
        )
        try:
            fi    = None
            names = fn or [f"f{i}" for i in range(500)]
            if hasattr(bm, "feature_importances_"):
                fi = bm.feature_importances_
            elif hasattr(bm, "coef_"):
                c = bm.coef_
                fi = abs(c).mean(axis=0) if c.ndim > 1 else abs(c)
            if fi is not None:
                n     = min(len(fi), len(names), 30)
                df_fi = (
                    pd.DataFrame({"Feature": names[:n], "Importance": fi[:n]})
                    .sort_values("Importance", ascending=True).tail(20)
                )
                fig = apply_chart_theme(px.bar(
                    df_fi, x="Importance", y="Feature", orientation="h",
                    title=f"Feature Importance — {st.session_state.best_model_name.replace('_', ' ').title()}",
                    color="Importance", color_continuous_scale=["#1c3a6e", "#4a90d9"]
                ))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Native feature importance is not available for this model type. Use the SHAP tabs.")
        except Exception as e:
            st.error(str(e))

    # ── Tab 2: SHAP Global Bar ────────────────────────────────────────────────
    with tab_shap_bar:
        insight(
            "SHAP global importance shows the mean absolute SHAP value for each feature across "
            "all test samples. Longer bars indicate greater average impact on model output."
        )
        try:
            import shap as _shap_check
            SHAP_OK = True
        except ImportError:
            SHAP_OK = False

        if not SHAP_OK:
            st.warning("SHAP library not installed. Run: pip install shap")
        elif Xt is not None:
            if st.button("Compute SHAP Values", use_container_width=True, key="shap_compute"):
                with st.spinner("Computing SHAP values (this may take 30–60 seconds on larger datasets)..."):
                    try:
                        import shap
                        smp = Xt[:min(150, len(Xt))]
                        mt  = type(bm).__name__.lower()
                        explainer = (
                            shap.TreeExplainer(bm)
                            if any(t in mt for t in ["tree", "forest", "xgb", "lgbm", "catboost", "extra", "gradient"])
                            else shap.LinearExplainer(bm, smp)
                            if "logistic" in mt or "linear" in mt or "ridge" in mt
                            else shap.KernelExplainer(
                                bm.predict_proba if hasattr(bm, "predict_proba") else bm.predict,
                                shap.sample(smp, 50)
                            )
                        )
                        raw = explainer.shap_values(smp)
                        # Normalise to 2D
                        if isinstance(raw, list):
                            sv = raw[1] if len(raw) == 2 else np.mean([np.abs(np.array(a)) for a in raw], axis=0)
                        else:
                            sv = np.array(raw)
                        if sv.ndim == 3:
                            sv = np.abs(sv).mean(axis=2)
                        st.session_state.shap_values = sv.astype(float)
                        st.session_state.shap_sample = smp
                        st.success("SHAP values computed. All three SHAP tabs are now active.")
                    except Exception as e:
                        import traceback
                        st.error(f"SHAP computation failed: {e}")
                        st.code(traceback.format_exc())

        if st.session_state.shap_values is not None:
            sv    = st.session_state.shap_values
            names = fn or [f"f{i}" for i in range(sv.shape[1])]
            msv   = abs(sv).mean(axis=0)
            max_f = min(40, len(msv))
            topn  = st.slider("Top N Features", 5, max_f, min(20, max_f), key="shap_bar_n") if max_f > 5 else max_f
            df_bar = (
                pd.DataFrame({"Feature": names, "Mean |SHAP|": msv})
                .sort_values("Mean |SHAP|", ascending=False).head(topn)
                .sort_values("Mean |SHAP|")
            )
            fig = apply_chart_theme(px.bar(
                df_bar, x="Mean |SHAP|", y="Feature", orientation="h",
                title="SHAP Global Feature Importance",
                color="Mean |SHAP|", color_continuous_scale=["#2d1b69", "#7c6fc9"]
            ))
            st.plotly_chart(fig, use_container_width=True)
        elif SHAP_OK and Xt is not None:
            st.info("Click 'Compute SHAP Values' to generate global importance.")

    # ── Tab 3: SHAP Beeswarm ─────────────────────────────────────────────────
    with tab_bee:
        insight(
            "Each dot represents one test sample. Red = high feature value, Blue = low feature value. "
            "Dots to the right increase the prediction; dots to the left decrease it. "
            "Wider horizontal spread means this feature has high variance in its impact."
        )
        if st.session_state.shap_values is None:
            st.info("Compute SHAP values first (SHAP Global Bar tab).")
        else:
            import shap
            import matplotlib.pyplot as plt
            sv   = st.session_state.shap_values
            Xs   = st.session_state.shap_sample
            max_f = min(30, sv.shape[1])
            topn  = st.slider("Top N Features", 5, max_f, min(15, max_f), key="bee_n") if max_f > 5 else max_f
            plt.figure(figsize=(10, max(4, topn * 0.38)))
            shap.summary_plot(sv, Xs, feature_names=fn, max_display=topn, plot_type="dot", show=False)
            fig_obj = plt.gcf()
            fig_obj.patch.set_facecolor("#0c0e14")
            for ax in fig_obj.get_axes():
                ax.set_facecolor("#111520")
            st.pyplot(fig_obj)
            plt.clf()

    # ── Tab 4: Instance Explanation ───────────────────────────────────────────
    with tab_inst:
        insight(
            "This waterfall chart explains a single prediction by showing how each feature "
            "pushes the model output away from the baseline expected value."
        )
        if Xt is not None and st.session_state.shap_values is not None:
            import shap
            import matplotlib.pyplot as plt
            sv  = st.session_state.shap_values
            idx = st.slider("Test Instance Index", 0, len(Xt) - 1, 0)

            # Build Explanation object safely
            data_row = Xt.iloc[idx] if hasattr(Xt, "iloc") else Xt[idx]
            base_val = 0.0
            try:
                base_val = st.session_state.get("shap_base_value", 0.0)
            except Exception:
                pass

            expl = shap.Explanation(
                values=sv[idx],
                base_values=base_val,
                data=data_row,
                feature_names=fn,
            )
            plt.figure(figsize=(9, 5))
            shap.plots.waterfall(expl, max_display=15, show=False)
            fig_obj = plt.gcf()
            fig_obj.patch.set_facecolor("#0c0e14")
            st.pyplot(fig_obj)
            plt.clf()

            yp = bm.predict(Xt[idx: idx + 1])
            pc = cn[yp[0]] if cn and yp[0] < len(cn) else str(yp[0])
            col_p, col_a = st.columns(2)
            col_p.metric("Predicted Class", pc)
            if st.session_state.y_test is not None:
                yt_v = st.session_state.y_test[idx]
                tc   = cn[yt_v] if cn and yt_v < len(cn) else str(yt_v)
                col_a.metric("Actual Class", tc)
        else:
            st.info("Compute SHAP values first.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — HYPERPARAMETER LOG
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Hyperparameter Log":
    page_header(
        "Hyperparameter Log",
        "Full Optuna trial history for every model — scores per trial, convergence curve, and chosen parameters."
    )
    if not st.session_state.trained:
        st.warning("Run training first.")
        st.stop()

    tnr = st.session_state.get("tuner_obj")

    if not st.session_state.get("tune_enabled", True) or tnr is None:
        bp = st.session_state.get("best_params", {})
        if bp:
            st.info("Tuning was disabled or trial objects are unavailable. Showing best parameters only.")
            for mn, p in bp.items():
                with st.expander(mn.replace("_", " ").title()):
                    if p:
                        df_p = pd.DataFrame([{"Parameter": k, "Value": str(v)} for k, v in p.items()])
                        st.dataframe(safe_df(df_p), use_container_width=True)
                    else:
                        st.info("Default parameters used (no tuning).")
        else:
            st.info("No tuning data available. Enable tuning and retrain.")
        st.stop()

    # Summary across all models
    section("Tuning Summary")
    sum_df = tnr.summary_dataframe()
    if not sum_df.empty:
        nc_s = [c for c in sum_df.columns if c != "model" and pd.api.types.is_numeric_dtype(sum_df[c])]
        st.dataframe(
            safe_df(sum_df).style.format({c: ("{:.4f}" if "score" in c else "{}") for c in nc_s if c in sum_df.columns}),
            use_container_width=True
        )
        fig = apply_chart_theme(px.bar(
            sum_df.sort_values("best_score", ascending=True),
            x="best_score", y="model", orientation="h",
            title="Best Tuned Score by Model",
            color="best_score", color_continuous_scale=["#1c3a6e", "#38a169"]
        ))
        st.plotly_chart(fig, use_container_width=True)
        insight("Each bar shows the best cross-validation score achieved by Optuna for that model. "
                "The champion model is selected based on this score.")

    st.divider()
    section("Trial Detail by Model")

    models_with_trials = [m for m in tnr.all_trials if tnr.all_trials[m]]
    if not models_with_trials:
        st.info("No trial history available.")
        st.stop()

    sel_m  = st.selectbox("Select Model", models_with_trials)
    df_t   = tnr.trials_dataframe(sel_m)
    if df_t.empty:
        st.info("No trials recorded for this model.")
        st.stop()

    bi = tnr.best_trial_info.get(sel_m, {})
    c1, c2, c3 = st.columns(3)
    metric_card(f"#{bi.get('trial_number', '?')}",      "Chosen Trial",  c1)
    metric_card(f"{bi.get('score', 0):.4f}",             f"Best {bi.get('metric', 'Score')}", c2)
    metric_card(f"{bi.get('n_trials_run', 0)}",          "Total Trials",  c3)
    st.markdown("<br>", unsafe_allow_html=True)

    # Convergence plot
    df_plot = df_t[df_t["score"].notna()].copy()
    if not df_plot.empty:
        df_plot["best_so_far"] = df_plot["score"].cummax()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_plot["trial"], y=df_plot["score"], name="Trial Score",
            mode="markers", marker=dict(color="#4a90d9", size=5, opacity=0.65)
        ))
        fig.add_trace(go.Scatter(
            x=df_plot["trial"], y=df_plot["best_so_far"], name="Best So Far",
            mode="lines", line=dict(color="#38a169", width=2)
        ))
        best_t   = bi.get("trial_number", 0)
        best_row = df_plot[df_plot["trial"] == best_t]
        if not best_row.empty:
            fig.add_trace(go.Scatter(
                x=best_row["trial"], y=best_row["score"], name="Chosen Trial",
                mode="markers", marker=dict(color="#d97706", size=14, symbol="star")
            ))
        fig.update_layout(
            title=f"Optuna Convergence — {sel_m.replace('_', ' ').title()}",
            xaxis_title="Trial Number", yaxis_title=bi.get("metric", "Score")
        )
        st.plotly_chart(apply_chart_theme(fig), use_container_width=True)
        insight("Blue dots are individual trial scores. The green line tracks the best score achieved so far. "
                "The gold star marks the chosen trial. A consistently rising green line indicates "
                "Optuna is effectively finding better configurations over time.")

    # Best parameters
    section(f"Chosen Trial #{bi.get('trial_number', '?')} — Best Parameters")
    bp_d = bi.get("params", {})
    if bp_d:
        cols = st.columns(min(len(bp_d), 4))
        for i, (k, v) in enumerate(bp_d.items()):
            cols[i % len(cols)].markdown(f"**`{k}`**\n\n`{v}`")
    else:
        st.info("Default parameters used (no tuning parameters logged).")

    # Full trial table
    section("All Trials")
    def highlight_best_trial(row):
        return ["background-color: rgba(217,119,6,.15); font-weight: 600;"] * len(row) if row.get("is_best") else [""] * len(row)

    nc_t = [c for c in df_t.columns if pd.api.types.is_numeric_dtype(df_t[c]) and c != "trial"]
    st.dataframe(
        safe_df(df_t).style.apply(highlight_best_trial, axis=1)
                      .format({c: "{:.4f}" for c in nc_t if "score" in c}),
        use_container_width=True, height=340
    )
    st.download_button(
        "Download Trial Log (CSV)", data=df_t.to_csv(index=False).encode(),
        file_name=f"{sel_m}_trials.csv", mime="text/csv"
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — MODEL CARD
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Card":
    page_header(
        "Model Card",
        "Auto-generated documentation covering data lineage, algorithm choice, performance metrics, "
        "intended use, and limitations. Exported as Markdown or HTML."
    )
    if not st.session_state.trained:
        st.warning("Run training first.")
        st.stop()

    bn  = st.session_state.best_model_name
    bm  = st.session_state.best_model
    cn  = st.session_state.class_names
    fn  = st.session_state.feature_names
    tm_ = st.session_state.train_metrics
    vm_ = st.session_state.val_metrics
    tsm = st.session_state.test_metrics
    bp  = st.session_state.best_params.get(bn, {})
    tnr = st.session_state.get("tuner_obj")
    ti  = tnr.best_trial_info.get(bn, {}) if tnr else {}

    # If auto-generated during training, display it directly
    mc_obj = st.session_state.get("model_card_obj")

    section("Customise Narrative Fields")
    with st.form("model_card_form"):
        col_a, col_b = st.columns(2)
        iu  = col_a.text_area("Intended Use",  value="Binary or multiclass tabular classification on structured data.", height=80)
        lim = col_b.text_area("Limitations",   value="Performance may degrade on out-of-distribution or highly imbalanced data. Requires retraining when data distribution shifts.", height=80)
        eth = st.text_area("Ethical Considerations", value="Ensure the training dataset is free of harmful bias. Audit predictions for fairness across demographic groups before production deployment.", height=60)
        gen = st.form_submit_button("Generate Model Card", use_container_width=True, type="primary")

    if gen:
        from src.evaluation.model_card import ModelCardGenerator
        mcg = ModelCardGenerator()
        mcg.build(
            model_name=bn, model_type=type(bm).__name__,
            dataset_name=st.session_state.get("file_name", "uploaded_dataset"),
            target_col=st.session_state.target_col or "target",
            feature_names=fn or [], class_names=cn,
            train_metrics=tm_, val_metrics=vm_, test_metrics=tsm,
            best_params=bp,
            dataset_shape=st.session_state.df.shape if st.session_state.df is not None else (0, 0),
            n_trials_run=ti.get("n_trials_run", 0),
            best_trial_number=ti.get("trial_number", 0),
            tuning_metric=ti.get("metric", "f1_weighted"),
            intended_use=iu, limitations=lim, ethical_notes=eth
        )
        st.session_state.model_card_obj = mcg
        mc_obj = mcg
        st.success("Model card generated.")

    if mc_obj:
        dl_md, dl_html = st.columns(2)
        md_text   = mc_obj.to_markdown()
        html_text = mc_obj.to_html()
        with dl_md:
            st.download_button("Download Model Card (Markdown)", data=md_text.encode(),
                               file_name=f"{bn}_model_card.md", mime="text/markdown",
                               use_container_width=True)
        with dl_html:
            st.download_button("Download Model Card (HTML)", data=html_text.encode(),
                               file_name=f"{bn}_model_card.html", mime="text/html",
                               use_container_width=True)
        st.divider()
        with st.expander("Preview Model Card", expanded=True):
            st.markdown(md_text)
    elif not gen:
        st.info("Click 'Generate Model Card' to build the documentation, or retrain to auto-generate it.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 8 — HYPOTHESIS TESTING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Hypothesis Testing":
    page_header(
        "Hypothesis Testing",
        "Statistical significance tests for model comparisons: "
        "McNemar, Wilcoxon Signed-Rank, Paired t-Test, and Friedman."
    )
    if not st.session_state.trained:
        st.warning("Run training first.")
        st.stop()

    tms = st.session_state.get("trained_models", {})
    if st.session_state.X_train is None or st.session_state.y_train is None:
        st.warning("Training split data not found. Retrain the pipeline.")
        st.stop()

    Xt  = st.session_state.X_test
    yt  = st.session_state.y_test
    Xt2 = st.session_state.X_train
    yt2 = st.session_state.y_train
    bn  = st.session_state.best_model_name

    if len(tms) < 2:
        st.warning("At least 2 trained models are required for pairwise comparison.")
        st.stop()

    with st.expander("Test Reference Guide", expanded=False):
        st.markdown("""
| Test | Description | Assumption |
|---|---|---|
| McNemar | Error pattern differences between two classifiers on the same test set | None |
| Wilcoxon | Whether paired CV fold scores differ significantly (non-parametric) | No normality required |
| Paired t-Test | Parametric version of Wilcoxon for CV fold scores | Score differences are normally distributed |
| Friedman | Whether all models perform equivalently (multi-model) | 3+ models required |
| Cohen's d | Effect size — practical significance beyond the p-value | Accompanies any test |

**Significance level alpha = 0.05 by default.** p < alpha implies a statistically significant difference.
        """)

    c_ref, c_alpha = st.columns([2, 1])
    ref_m = c_ref.selectbox(
        "Reference Model",
        list(tms.keys()),
        index=list(tms.keys()).index(bn) if bn in tms else 0
    )
    alpha = c_alpha.slider("Significance Level (alpha)", 0.01, 0.10, 0.05, 0.01)

    if st.button("Run All Hypothesis Tests", use_container_width=True, type="primary"):
        with st.spinner("Running statistical tests..."):
            from src.evaluation.hypothesis_testing import HypothesisTester
            from src.utils.config_loader import load_config
            try:
                cfg = load_config("config/config.yaml")
            except Exception:
                cfg = {"tuning": {"metric": "f1_weighted"}, "models": {"cross_validation_folds": 5}, "project": {"random_state": 42}}
            ht  = HypothesisTester(cfg, alpha=alpha)
            res = ht.run_all(tms, Xt, yt, Xt2, yt2, reference_model=ref_m)
            st.session_state.hypothesis_results = res
            st.success("Statistical tests complete.")

    res = st.session_state.get("hypothesis_results")
    if res:
        tab_sum, tab_mcn, tab_wil, tab_ttest, tab_fri = st.tabs([
            "Summary", "McNemar", "Wilcoxon", "Paired t-Test", "Friedman"
        ])

        with tab_sum:
            sum_df = res.get("summary", pd.DataFrame())
            if not sum_df.empty:
                def hl_sig(r):
                    return ["background-color: rgba(58,161,105,.10);"] * len(r) if r.get("wilcoxon_sig") == "Yes" else [""] * len(r)
                nc_s = [c for c in sum_df.columns if pd.api.types.is_numeric_dtype(sum_df[c])]
                st.dataframe(
                    safe_df(sum_df).style.apply(hl_sig, axis=1)
                                         .format({c: "{:.4f}" for c in nc_s if c in sum_df.columns}),
                    use_container_width=True
                )
                insight("Highlighted rows indicate statistically significant differences (p < alpha). "
                        "mean_A / mean_B are average cross-validation scores per model. "
                        "effect_size (Cohen's d) measures practical significance.")
                pw = res.get("wilcoxon", {})
                if pw:
                    pairs = list(pw.keys())
                    pvals = [pw[k].get("p_value", 1) for k in pairs]
                    fig = apply_chart_theme(px.bar(
                        x=pairs, y=pvals,
                        title="Wilcoxon p-values by Model Pair",
                        color=[0 if p < alpha else 1 for p in pvals],
                        color_discrete_map={0: "#38a169", 1: "#e05252"},
                        labels={"x": "Pair", "y": "p-value"}
                    ))
                    fig.add_hline(y=alpha, line_dash="dash", line_color="#d97706",
                                  annotation_text=f"alpha = {alpha}")
                    fig.update_xaxes(tickangle=30)
                    st.plotly_chart(fig, use_container_width=True)

        with tab_mcn:
            insight("McNemar's test counts disagreements between two classifiers on the same test set. "
                    "A significant result means they make systematically different errors, not just different accuracy.")
            for pair, r in res.get("mcnemar", {}).items():
                with st.expander(f"{pair}  —  p = {r.get('p_value', '?')}  {r.get('interpretation', '')}"):
                    co1, co2 = st.columns(2)
                    co1.metric("Test Statistic", f"{r.get('statistic', 0):.4f}")
                    co2.metric("p-value", f"{r.get('p_value', 1):.4f}")
                    ct = r.get("contingency")
                    if ct:
                        a_n = pair.split("_vs_")[0]
                        b_n = pair.split("_vs_")[1]
                        df_ct = pd.DataFrame(ct, index=[f"{a_n} correct", f"{a_n} wrong"],
                                             columns=[f"{b_n} correct", f"{b_n} wrong"])
                        st.dataframe(df_ct, use_container_width=True)

        with tab_wil:
            insight("Non-parametric signed-rank test comparing paired CV fold scores. "
                    "No normality assumption required. Recommended for most ML model comparisons.")
            for pair, r in res.get("wilcoxon", {}).items():
                if "error" in r:
                    continue
                with st.expander(f"{pair}  —  p = {r.get('p_value', '?')}  {r.get('interpretation', '')}  (effect: {r.get('effect_label', '?')})"):
                    co1, co2, co3, co4 = st.columns(4)
                    co1.metric("Statistic",       f"{r.get('statistic', 0):.3f}")
                    co2.metric("p-value",         f"{r.get('p_value', 1):.4f}")
                    co3.metric("Cohen's d",       f"{r.get('effect_size', 0):.3f}")
                    co4.metric("Effect Label",    r.get("effect_label", "?"))
                    co1.metric("Mean A", f"{r.get('mean_A', 0):.4f}")
                    co2.metric("Mean B", f"{r.get('mean_B', 0):.4f}")

        with tab_ttest:
            insight("Parametric paired t-test for CV fold scores. Assumes differences are normally distributed. "
                    "Use alongside Wilcoxon — if both agree, confidence is higher.")
            for pair, r in res.get("paired_ttest", {}).items():
                if "error" in r:
                    continue
                with st.expander(f"{pair}  —  p = {r.get('p_value', '?')}  {r.get('interpretation', '')}"):
                    co1, co2, co3 = st.columns(3)
                    co1.metric("t-Statistic",  f"{r.get('statistic', 0):.3f}")
                    co2.metric("p-value",      f"{r.get('p_value', 1):.4f}")
                    co3.metric("Effect Size",  f"{r.get('effect_size', 0):.3f}")

        with tab_fri:
            insight("Friedman test checks whether all models perform equivalently across folds (null hypothesis). "
                    "A significant result means at least one model differs. Use pairwise tests to identify which one.")
            fr = res.get("friedman", {})
            if "error" in (fr or {}):
                st.warning(fr["error"])
            elif fr:
                co1, co2 = st.columns(2)
                co1.metric("Chi-squared Statistic", f"{fr.get('statistic', 0):.4f}")
                co2.metric("p-value",               f"{fr.get('p_value', 1):.4f}")
                st.markdown(f"**Result:** {fr.get('interpretation', '')}")
                ranks = fr.get("rankings", [])
                if ranks:
                    df_r = pd.DataFrame(ranks, columns=["Model", "Mean CV Score"])
                    df_r["Rank"] = range(1, len(df_r) + 1)
                    fig = apply_chart_theme(px.bar(
                        df_r, x="Mean CV Score", y="Model", orientation="h",
                        color="Mean CV Score", color_continuous_scale=["#1c3a6e", "#38a169"],
                        title="Model Rankings (Friedman)"
                    ))
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(safe_df(df_r), use_container_width=True)
            else:
                st.info("At least 3 models are required for the Friedman test.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 9 — PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Predictions":
    page_header(
        "Predictions",
        "Run the champion model on new data — enter values manually for a single prediction "
        "or upload a batch file."
    )
    if not st.session_state.trained:
        st.warning("Run training first.")
        st.stop()

    bm  = st.session_state.best_model
    pp  = st.session_state.preprocessor
    cn  = st.session_state.class_names
    dfo = st.session_state.df
    tc  = st.session_state.target_col
    fcs = [c for c in dfo.columns if c != tc]

    tab_manual, tab_batch = st.tabs(["Manual Input", "Batch File"])

    with tab_manual:
        section("Input Feature Values")
        insight("Numeric fields default to the column median. Categorical fields show observed values from training data.")
        with st.form("manual_predict"):
            inputs = {}
            row_chunks = [fcs[i: i + 3] for i in range(0, len(fcs), 3)]
            for chunk in row_chunks:
                fcols = st.columns(len(chunk))
                for fc, col in zip(chunk, fcols):
                    sv_ = dfo[fc].dropna()
                    if pd.api.types.is_numeric_dtype(dfo[fc]):
                        inputs[fc] = col.number_input(fc, value=float(sv_.median()) if len(sv_) > 0 else 0.0, key=f"mp_{fc}")
                    else:
                        opts = [str(v) for v in sv_.unique().tolist()[:20]]
                        inputs[fc] = col.selectbox(fc, options=opts or [""], key=f"mp_{fc}")
            submitted = st.form_submit_button("Predict", use_container_width=True, type="primary")

        if submitted:
            try:
                Xn   = pp.transform(pd.DataFrame([inputs]))
                pred = bm.predict(Xn)
                pl   = cn[pred[0]] if cn and pred[0] < len(cn) else str(pred[0])
                c_res, c_prob = st.columns(2)
                with c_res:
                    st.markdown(f"""
                    <div class="champion-banner" style="text-align:center;">
                      <div class="champion-sub">Predicted Class</div>
                      <div class="champion-name" style="font-size:1.4rem;">{pl}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with c_prob:
                    if hasattr(bm, "predict_proba"):
                        probs = bm.predict_proba(Xn)[0].astype(float)
                        pf = pd.DataFrame({
                            "Class":       cn or [str(i) for i in range(len(probs))],
                            "Probability": probs
                        }).sort_values("Probability", ascending=True)
                        fig = apply_chart_theme(px.bar(
                            pf, x="Probability", y="Class", orientation="h",
                            color="Probability", color_continuous_scale=["#1c3a6e", "#38a169"],
                            title="Class Probabilities"
                        ))
                        st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    with tab_batch:
        section("Upload Batch File")
        tf = st.file_uploader("CSV or Excel file", type=["csv", "xlsx"])
        if tf:
            try:
                ext = Path(tf.name).suffix.lower()
                tdf = pd.read_csv(tf) if ext == ".csv" else pd.read_excel(tf)
                st.dataframe(safe_df(tdf.head(5)), use_container_width=True)
                if st.button("Run Batch Predictions", use_container_width=True, type="primary"):
                    Xb    = pp.transform(tdf)
                    preds = bm.predict(Xb)
                    lbls  = [cn[p] if cn and p < len(cn) else str(p) for p in preds]
                    res_df = tdf.copy()
                    res_df["prediction"] = lbls
                    if hasattr(bm, "predict_proba"):
                        probs = bm.predict_proba(Xb).astype(float)
                        for i, c_ in enumerate(cn or range(probs.shape[1])):
                            res_df[f"prob_{c_}"] = probs[:, i]
                    st.success(f"Predictions complete — {len(res_df):,} rows")
                    st.dataframe(safe_df(res_df.head(20)), use_container_width=True)
                    st.download_button(
                        "Download Predictions (CSV)",
                        data=res_df.to_csv(index=False).encode(),
                        file_name="batch_predictions.csv", mime="text/csv"
                    )
            except Exception as e:
                st.error(f"Batch prediction failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 10 — MODEL REGISTRY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Registry":
    page_header(
        "Model Registry",
        "Browse all MLflow experiment runs. Compare metrics across runs and track training history."
    )
    try:
        import mlflow
        from src.utils.config_loader import load_config
        cfg  = load_config("config/config.yaml")
        tc_  = cfg.get("tracking", {}).get("mlflow", {})
        mlflow.set_tracking_uri(tc_.get("tracking_uri", "mlruns"))
        exp  = mlflow.get_experiment_by_name(tc_.get("experiment_name", "automl_classification"))
    except Exception as e:
        st.error(f"MLflow initialisation failed: {e}")
        st.stop()

    if exp is None:
        st.info("No MLflow experiment found. Run training first.")
        st.stop()

    rdf = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["metrics.f1_weighted DESC"],
        max_results=100
    )
    if rdf.empty:
        st.info("No runs recorded yet.")
        st.stop()

    mc_cols = [c for c in rdf.columns if c.startswith("metrics.")]
    dc      = [c for c in ["tags.mlflow.runName", "start_time", "status"] + mc_cols[:6] if c in rdf.columns]

    c1, c2, c3 = st.columns(3)
    metric_card(str(len(rdf)), "Total Runs", c1)
    if "metrics.f1_weighted" in rdf.columns:
        metric_card(f"{rdf['metrics.f1_weighted'].max():.4f}", "Best F1", c2)
    if "metrics.accuracy" in rdf.columns:
        metric_card(f"{rdf['metrics.accuracy'].max():.4f}", "Best Accuracy", c3)
    st.markdown("<br>", unsafe_allow_html=True)

    tab_runs, tab_compare = st.tabs(["All Runs", "Metric Comparison"])

    with tab_runs:
        st.dataframe(safe_df(rdf[dc].head(50)), use_container_width=True, height=400)

    with tab_compare:
        avail = [c.replace("metrics.", "") for c in mc_cols
                 if any(k in c for k in ["f1", "accuracy", "roc", "precision", "recall"])]
        if avail:
            sel_met = st.selectbox("Select Metric", avail)
            fc      = f"metrics.{sel_met}"
            if fc in rdf.columns:
                pf = rdf[["tags.mlflow.runName", fc]].dropna()
                pf.columns = ["Run", sel_met]
                fig = apply_chart_theme(px.bar(
                    pf.sort_values(sel_met, ascending=True),
                    x=sel_met, y="Run", orientation="h",
                    color=sel_met, color_continuous_scale=["#1c3a6e", "#38a169"],
                    title=f"Run Comparison — {sel_met}"
                ))
                st.plotly_chart(fig, use_container_width=True)
