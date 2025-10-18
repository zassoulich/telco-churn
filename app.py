import streamlit as st
import pandas as pd
import numpy as np
from typing import Tuple, List
from io import StringIO

# Modeling
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve

# Viz
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Telco Churn Dashboard",
    page_icon="📉",
    layout="wide",
)

# -----------------------------
# Helpers
# -----------------------------
DATA_DEFAULT_PATH = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
TARGET_COL = "Churn"
ID_COL = "customerID"

# Fixed color mapping for churn visualization
# No (not churned) = Blue, Yes (churned) = Red
CHURN_COLOR_MAP = {"No": "#1f77b4", "Yes": "#d62728"}

CATEGORICAL_COLS_DEFAULT = [
    "gender","SeniorCitizen","Partner","Dependents","PhoneService","MultipleLines",
    "InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport",
    "StreamingTV","StreamingMovies","Contract","PaperlessBilling","PaymentMethod"
]
NUMERIC_COLS_DEFAULT = ["tenure","MonthlyCharges","TotalCharges"]

# --- This function will preprocess the dataframe just like I did in the notebook ---
def preprocess_telco(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy() #better safe than sorry lol
    # Coerce TotalCharges to numeric vals (has empty spaces " " in raw so I replaced them with nan)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace({" ": np.nan}), errors="coerce")
    # SeniorCitizen as int/binary
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
    # Replace "No internet/phone service" → "No" in the columns listed below
    service_no_map_cols = [
        "MultipleLines","OnlineSecurity","OnlineBackup","DeviceProtection",
        "TechSupport","StreamingTV","StreamingMovies"
    ]
    for c in service_no_map_cols:
        if c in df.columns:
            df[c] = df[c].replace({"No internet service": "No", "No phone service": "No"})
    # Drop ID column if present
    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])
    # Drop rows in Target and other key (numeric) columns with missing numerics
    subset_cols = [TARGET_COL] if TARGET_COL in df.columns else []
    for c in ["tenure","MonthlyCharges","TotalCharges"]:
        if c in df.columns:
            subset_cols.append(c)
    if subset_cols:
        df = df.dropna(subset=subset_cols)
    # Strip trailing spaces in object columns
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].str.strip()
    return df

@st.cache_data(show_spinner=False)
def load_data(path: str = DATA_DEFAULT_PATH, uploaded: pd.DataFrame | None = None) -> pd.DataFrame:
    if uploaded is not None:
        df = uploaded.copy()
    else:
        df = pd.read_csv(path)
    # Clean known quirks of IBM Telco dataset
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace({" ": np.nan}), errors="coerce")
    if "SeniorCitizen" in df.columns:
        # sometimes stored as 0/1 integers; ensure string for categorical if in cat list
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
    # Drop rows where target is missing if present in dataframe
    if TARGET_COL in df.columns:
        df = df[~df[TARGET_COL].isna()]
    return df


def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    with st.sidebar:
        st.header("Filters")
        # Dynamic filters if columns available
        fdf = df.copy()
        if "Contract" in fdf.columns:
            contract = st.multiselect("Contract", sorted(fdf["Contract"].dropna().unique().tolist()))
            if contract:
                fdf = fdf[fdf["Contract"].isin(contract)]
        if "InternetService" in fdf.columns:
            internet = st.multiselect("Internet Service", sorted(fdf["InternetService"].dropna().unique().tolist()))
            if internet:
                fdf = fdf[fdf["InternetService"].isin(internet)]
        if "tenure" in fdf.columns:
            t_min, t_max = int(fdf["tenure"].min()), int(fdf["tenure"].max())
            tenure = st.slider("Tenure (months)", t_min, t_max, (t_min, t_max))
            fdf = fdf[(fdf["tenure"] >= tenure[0]) & (fdf["tenure"] <= tenure[1])]
        if "MonthlyCharges" in fdf.columns:
            m_min, m_max = float(fdf["MonthlyCharges"].min()), float(fdf["MonthlyCharges"].max())
            monthly = st.slider("Monthly Charges", m_min, m_max, (m_min, m_max))
            fdf = fdf[(fdf["MonthlyCharges"] >= monthly[0]) & (fdf["MonthlyCharges"] <= monthly[1])]
        return fdf


@st.cache_resource(show_spinner=False)
def build_model(cat_cols: List[str], num_cols: List[str]) -> Pipeline:
    # Preprocessor (fixes NaN issues on custom unporcessed CSVs)
    num_features = [c for c in num_cols if c in st.session_state.df.columns]
    cat_features = [c for c in cat_cols if c in st.session_state.df.columns]

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_features),
            ("cat", cat_pipe, cat_features),
        ],
        remainder="drop",
    )

    # Use class_weight balanced to handle churn imbalance
    model = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
    pipe = Pipeline(steps=[("prep", preprocessor), ("clf", model)])
    return pipe


def train_and_eval(df: pd.DataFrame, cat_cols: List[str], num_cols: List[str]) -> Tuple[Pipeline, dict]:
    # Mirror notebook-style preprocessing
    dfp = preprocess_telco(df)

    # Ensure numeric columns are numeric, coerce bad strings to NaN
    for c in ["tenure", "MonthlyCharges", "TotalCharges"]:
        if c in dfp.columns:
            dfp[c] = pd.to_numeric(dfp[c], errors="coerce")

    # Standardize target column encoding
    if TARGET_COL in dfp.columns:
        # Normalize string labels to consistent case
        if dfp[TARGET_COL].dtype == object:
            dfp[TARGET_COL] = dfp[TARGET_COL].str.strip().str.title()
            # Map Yes=1 (churned), No=0 (not churned)
            y = dfp[TARGET_COL].map({"Yes": 1, "No": 0})
        # Handle numeric encoding
        elif pd.api.types.is_numeric_dtype(dfp[TARGET_COL]):
            # Assume standard encoding: 1=churned, 0=not churned
            y = dfp[TARGET_COL].astype(int)
        else:
            raise ValueError("Unexpected target format")
        
        X = dfp.drop(columns=[TARGET_COL])
    else:
        raise ValueError("Target column 'Churn' not found")

    # Keep only columns that exist after preprocessing
    cat_cols = [c for c in cat_cols if c in X.columns]
    num_cols = [c for c in num_cols if c in X.columns]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Fit
    pipe = build_model(cat_cols, num_cols)
    pipe.fit(X_train, y_train)

    # Evaluate
    y_proba = pipe.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, y_proba)
    prauc = average_precision_score(y_test, y_proba)
    fpr, tpr, roc_thr = roc_curve(y_test, y_proba)
    prec, rec, pr_thr = precision_recall_curve(y_test, y_proba)

    metrics = {
        "roc_auc": roc,
        "pr_auc": prauc,
        "roc_curve": (fpr, tpr, roc_thr),
        "pr_curve": (prec, rec, pr_thr),
        "X_test": X_test,
        "y_test": y_test,
        "y_proba": y_proba,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
    }
    return pipe, metrics


def plot_roc(fpr, tpr):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Baseline", line=dict(dash="dash")))
    fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", title="ROC Curve")
    return fig


def plot_pr(prec, rec):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name="PR"))
    fig.update_layout(xaxis_title="Recall", yaxis_title="Precision", title="Precision-Recall Curve")
    return fig


def plot_confusion(cm, labels=("No", "Yes")):
    z = cm
    fig = go.Figure(data=go.Heatmap(z=z, x=[f"Pred {l}" for l in labels], y=[f"Actual {l}" for l in labels], text=z, texttemplate="%{text}", colorscale="Blues"))
    fig.update_layout(title="Confusion Matrix")
    return fig


# -----------------------------
# Sidebar data input
# -----------------------------
with st.sidebar:
    st.title("📉 Telco Churn")
    st.caption("Upload the IBM Telco churn CSV or use the default path.")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], accept_multiple_files=False)
    use_default = st.toggle("Use default repo path", value=True)

# Load data
try:
    df_uploaded = pd.read_csv(uploaded_file) if uploaded_file else None
except Exception as e:
    st.sidebar.error(f"Failed to read uploaded CSV: {e}")
    df_uploaded = None

st.session_state.df = load_data(uploaded=df_uploaded) if (uploaded_file is not None) else (load_data(DATA_DEFAULT_PATH) if use_default else None)

if st.session_state.get("df") is None:
    st.warning("No dataset loaded. Upload a CSV or enable 'Use default repo path'.")
    st.stop()

# Basic cleaning for UI convenience
df = st.session_state.get("df")
if df is None or df.empty:
    st.error("Dataset not loaded. Please upload a CSV or enable default path.")
    st.stop()
else:
    df = df.copy()



# -----------------------------
# Header
# -----------------------------
st.title("Telco Churn Insights Dashboard")
st.write("Interactive EDA, cohort analysis, and on-the-fly modeling.")

# -----------------------------
# Filters & KPIs
# -----------------------------
fdf = filter_dataframe(df)

left, mid, right, extra = st.columns(4)
with left:
    st.metric("Rows (after filters)", f"{len(fdf):,}")
with mid:
    churn_rate = (fdf[TARGET_COL].map({"Yes": 1, "No": 0}).mean() * 100) if TARGET_COL in fdf.columns else np.nan
    st.metric("Churn Rate", f"{churn_rate:.1f}%" if pd.notna(churn_rate) else "—")
with right:
    st.metric("Avg Monthly Charges", f"${fdf['MonthlyCharges'].mean():.2f}" if 'MonthlyCharges' in fdf.columns else "—")
with extra:
    st.metric("Avg Tenure (mo)", f"{fdf['tenure'].mean():.1f}" if 'tenure' in fdf.columns else "—")

# -----------------------------
# Tabs
# -----------------------------
TAB_OVERVIEW, TAB_COHORTS, TAB_MODEL, TAB_SCORE = st.tabs(["Overview", "Cohorts", "Modeling", "Score & Export CSV"])

with TAB_OVERVIEW:
    st.subheader("Distributions")
    cols_plot = st.multiselect(
        "Select columns to plot",
        options=[c for c in fdf.columns if c not in [ID_COL]],
        default=[c for c in ["Contract", "InternetService", "PaymentMethod", "MonthlyCharges", "tenure"] if c in fdf.columns]
    )

    name_map = {
    "Contract": "contract types",
    "InternetService": "internet service types",
    "PaymentMethod": "payment methods",
    "MonthlyCharges": "monthly charges",
    "tenure": "tenure (months)"
    }
    
    for col in cols_plot:
        if pd.api.types.is_numeric_dtype(fdf[col]):
            fig = px.histogram(
                fdf, x=col, 
                color=TARGET_COL if TARGET_COL in fdf.columns else None,
                color_discrete_map=CHURN_COLOR_MAP if TARGET_COL in fdf.columns else None,
                category_orders={TARGET_COL: ["No", "Yes"]} if TARGET_COL in fdf.columns else None,
                nbins=40, 
                marginal="box"
            )
        else:
            fig = px.histogram(
                fdf, x=col, 
                color=TARGET_COL if TARGET_COL in fdf.columns else None,
                color_discrete_map=CHURN_COLOR_MAP if TARGET_COL in fdf.columns else None,
                category_orders={TARGET_COL: ["No", "Yes"]} if TARGET_COL in fdf.columns else None
            )
        fig.update_layout(title=f"Distribution of {name_map.get(col, col)}",
                          xaxis_title=name_map.get(col, col))
        st.plotly_chart(fig, use_container_width=True)

with TAB_COHORTS:
    st.subheader("Churn by Contract / Internet Service")
    if all(c in fdf.columns for c in ["Contract", "InternetService", TARGET_COL]):
        temp = (
            fdf.groupby(["Contract", "InternetService"], dropna=False)[TARGET_COL]
            .apply(lambda s: (s.map({"Yes":1, "No":0}).mean() * 100))
            .reset_index(name="Churn %")
        )
        fig = px.bar(temp, x="Contract", y="Churn %", color="InternetService", barmode="group", text=temp["Churn %"].round(1))
        fig.update_layout(yaxis_title="Churn %")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Monthly Charges vs Tenure")
    if all(c in fdf.columns for c in ["MonthlyCharges","tenure"]):
        # Drop rows with NaN before fitting
        fdf_scatter = fdf.dropna(subset=["MonthlyCharges", "tenure"]).copy()
        
        if len(fdf_scatter) >= 3:
            fig = px.scatter(
                fdf_scatter,
                x="tenure",
                y="MonthlyCharges",
                color=TARGET_COL if TARGET_COL in fdf_scatter.columns else None,
                color_discrete_map=CHURN_COLOR_MAP if TARGET_COL in fdf_scatter.columns else None,
                category_orders={TARGET_COL: ["No", "Yes"]} if TARGET_COL in fdf_scatter.columns else None,
                hover_data=[ID_COL] if ID_COL in fdf_scatter.columns else None,
                trendline="ols",
                marginal_x="histogram",
                marginal_y="histogram",
                opacity=0.6
            )
        else:
            fig = px.scatter(
                fdf_scatter,
                x="tenure",
                y="MonthlyCharges",
                color=TARGET_COL if TARGET_COL in fdf_scatter.columns else None,
                color_discrete_map=CHURN_COLOR_MAP if TARGET_COL in fdf_scatter.columns else None,
                category_orders={TARGET_COL: ["No", "Yes"]} if TARGET_COL in fdf_scatter.columns else None,
                hover_data=[ID_COL] if ID_COL in fdf_scatter.columns else None,
            )
        
        fig.update_layout(
            xaxis_title="Tenure (months)", 
            yaxis_title="Monthly Charges ($)",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

with TAB_MODEL:
    st.subheader("Train a quick model (Logistic Regression)")
    st.caption("For production, train offline and load a persisted model. This demo trains on the fly.")

    cat_cols = st.multiselect("Categorical columns", options=[c for c in CATEGORICAL_COLS_DEFAULT if c in df.columns], default=[c for c in CATEGORICAL_COLS_DEFAULT if c in df.columns])
    num_cols = st.multiselect("Numeric columns", options=[c for c in NUMERIC_COLS_DEFAULT if c in df.columns], default=[c for c in NUMERIC_COLS_DEFAULT if c in df.columns])

    if st.button("Train / Re-train", type="primary"):
        with st.spinner("Training model..."):
            pipe, metrics = train_and_eval(df, cat_cols, num_cols)
            st.session_state.model = pipe
            st.session_state.metrics = metrics

    if "metrics" in st.session_state:
        m = st.session_state.metrics
        st.success(f"ROC-AUC: {m['roc_auc']:.3f} · PR-AUC: {m['pr_auc']:.3f}")
        col1, col2 = st.columns(2)
        with col1:
            fpr, tpr, _ = m["roc_curve"]
            st.plotly_chart(plot_roc(fpr, tpr), use_container_width=True)
        with col2:
            prec, rec, _ = m["pr_curve"]
            st.plotly_chart(plot_pr(prec, rec), use_container_width=True)

        st.subheader("Threshold & Confusion Matrix")
        thr = st.slider("Prediction threshold", 0.0, 1.0, 0.30, 0.01)
        y_pred = (m["y_proba"] >= thr).astype(int)
        cm = confusion_matrix(m["y_test"], y_pred)
        st.plotly_chart(plot_confusion(cm), use_container_width=True)

with TAB_SCORE:
    st.subheader("Score current (filtered) dataset")
    if "model" not in st.session_state:
        st.info("Train a model in the 'Modeling' tab first.")
    else:
        mdl: Pipeline = st.session_state.model
        # Drop target and ID from features before predicting
        X_score = fdf.drop(columns=[TARGET_COL, ID_COL], errors="ignore")
        proba = mdl.predict_proba(X_score)[:, 1]
        thr2 = st.slider("Threshold for flags", 0.0, 1.0, 0.30, 0.01, key="thr2")
        pred = (proba >= thr2).astype(int)

        # Build scored output starting from ID/target (when available) to avoid duplicate errors
        base_cols = []
        if ID_COL in fdf.columns:
            base_cols.append(ID_COL)
        if TARGET_COL in fdf.columns:
            base_cols.append(TARGET_COL)
        scored = fdf[base_cols].copy() if base_cols else pd.DataFrame(index=fdf.index)
        scored["churn_proba"] = proba
        scored["churn_pred"] = pred

        st.dataframe(scored.head(50), use_container_width=True)

        csv = scored.to_csv(index=False)
        st.download_button("Download scored CSV", data=csv, file_name="scored_telco_churn.csv", mime="text/csv")

# Footer note
st.caption("Tip: Save a trained model with joblib and load it in this app for faster scoring in production.")