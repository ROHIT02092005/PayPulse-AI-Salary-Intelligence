import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import math, io, os

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PayPulse — AI Salary Intelligence",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060b18 0%, #0f172a 60%, #1a1040 100%);
    border-right: 1px solid #1e2d4a;
}
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
section[data-testid="stSidebar"] .stRadio label {
    font-size: 0.88rem; padding: 5px 0; letter-spacing: 0.01em;
}

/* ── Main background ── */
.stApp { background: #060b18; }
.block-container { padding: 1.5rem 2.5rem 4rem; }

/* ── KPI cards ── */
.kpi-card {
    background: linear-gradient(135deg, #111827, #0d1525);
    border: 1px solid #1e2d4a;
    border-radius: 16px;
    padding: 18px 20px;
    text-align: center;
    transition: transform .2s, border-color .2s, box-shadow .2s;
}
.kpi-card:hover {
    transform: translateY(-4px);
    border-color: #6366f1;
    box-shadow: 0 8px 32px rgba(99,102,241,0.18);
}
.kpi-label { font-size: 0.7rem; color: #64748b; text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 6px; }
.kpi-value { font-size: 1.8rem; font-weight: 800; color: #818cf8; }
.kpi-sub   { font-size: 0.75rem; color: #475569; margin-top: 3px; }

/* ── Prediction box ── */
.pred-box {
    background: linear-gradient(135deg, #052e16, #0b3a1e);
    border: 2px solid #22c55e;
    border-radius: 20px;
    padding: 28px 24px;
    text-align: center;
    box-shadow: 0 0 48px rgba(34,197,94,0.18);
}
.pred-label  { font-size: 0.78rem; color: #86efac; letter-spacing: 2.5px; text-transform: uppercase; margin-bottom: 6px; }
.pred-salary { font-size: 3.2rem; font-weight: 900; color: #22c55e; line-height: 1; }
.pred-range  { font-size: 0.85rem; color: #4ade80; margin-top: 8px; }

/* ── Placeholder box ── */
.placeholder-box {
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    height: 340px;
    border: 2px dashed #1e2d4a;
    border-radius: 20px;
    background: rgba(15,23,42,0.4);
}

/* ── Tip card ── */
.tip-card {
    background: rgba(251,191,36,0.07);
    border-left: 4px solid #f59e0b;
    border-radius: 10px;
    padding: 11px 15px;
    margin: 7px 0;
    font-size: 0.85rem;
    color: #fde68a;
    line-height: 1.5;
}

/* ── Section titles ── */
.section-title { font-size: 1.5rem; font-weight: 800; color: #f1f5f9; margin-bottom: 4px; }
.section-sub   { font-size: 0.88rem; color: #475569; margin-bottom: 24px; }

/* ── Compare card ── */
.cmp-card { background: #111827; border: 1px solid #1e2d4a; border-radius: 14px; padding: 16px; }
.cmp-name { font-size: 0.95rem; font-weight: 700; color: #e2e8f0; margin-bottom: 10px; }

/* ── Buttons ── */
div.stButton > button {
    background: linear-gradient(90deg, #4f46e5, #7c3aed) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; padding: 10px 28px !important;
    font-weight: 700 !important; font-size: 0.9rem !important;
    width: 100% !important; transition: all .2s !important;
    letter-spacing: 0.02em !important;
}
div.stButton > button:hover {
    background: linear-gradient(90deg, #4338ca, #6d28d9) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(124,58,237,0.35) !important;
}

/* ── Animations ── */
@keyframes pulse-ring {
    0%   { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(99,102,241,0.5); }
    70%  { transform: scale(1);    box-shadow: 0 0 0 14px rgba(99,102,241,0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(99,102,241,0); }
}
@keyframes float {
    0%,100% { transform: translateY(0px); }
    50%      { transform: translateY(-8px); }
}
@keyframes shimmer {
    0%   { background-position: -200% center; }
    100% { background-position:  200% center; }
}
@keyframes fade-in-up {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes ticker {
    0%   { transform: translateX(0); }
    100% { transform: translateX(-50%); }
}
@keyframes orbit {
    from { transform: rotate(0deg) translateX(38px) rotate(0deg); }
    to   { transform: rotate(360deg) translateX(38px) rotate(-360deg); }
}

.paypulse-logo-pulse {
    animation: pulse-ring 2.2s ease infinite;
    display: inline-block; border-radius: 50%;
}
.paypulse-float { animation: float 3.5s ease-in-out infinite; }
.paypulse-shimmer {
    background: linear-gradient(90deg, #818cf8, #c084fc, #f472b6, #818cf8);
    background-size: 200% auto;
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shimmer 3s linear infinite;
}
.fade-in-up { animation: fade-in-up 0.7s ease both; }

.ticker-wrap {
    overflow: hidden; background: rgba(99,102,241,0.08);
    border-top: 1px solid #1e2d4a; border-bottom: 1px solid #1e2d4a;
    padding: 6px 0; margin: 12px 0 20px;
}
.ticker {
    display: inline-flex; white-space: nowrap;
    animation: ticker 22s linear infinite;
    font-size: 0.78rem; color: #818cf8;
}
.ticker span { margin: 0 32px; }

.orbit-dot {
    width: 8px; height: 8px;
    background: #818cf8; border-radius: 50%;
    position: absolute;
    animation: orbit 3s linear infinite;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────
def hex_to_rgba(hex_color: str, alpha: float = 0.15) -> str:
    """Convert #rrggbb → rgba(r,g,b,alpha) — safe for Plotly fillcolor."""
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'


def predict(model, scaler, age, gender, edu, job, exp, le_dict):
    g = le_dict['Gender'].transform([gender])[0]
    e = le_dict['Education Level'].transform([edu])[0]
    j = le_dict['Job Title'].transform([job])[0]
    X = np.array([[age, g, e, j, exp]])
    return max(0.0, float(model.predict(scaler.transform(X))[0]))


def fmt(val):
    return f"${val:,.0f}"


def plotly_dark(fig, height=420):
    fig.update_layout(
        height=height,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(10,15,30,0.8)',
        font=dict(color='#94a3b8', family='Inter'),
        title_font=dict(color='#f1f5f9', size=15, family='Inter'),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#94a3b8')),
        xaxis=dict(gridcolor='#111827', linecolor='#1e2d4a', zerolinecolor='#1e2d4a'),
        yaxis=dict(gridcolor='#111827', linecolor='#1e2d4a', zerolinecolor='#1e2d4a'),
        margin=dict(t=48, r=16, b=40, l=16)
    )
    return fig


# ─────────────────────────────────────────────────────────────────
# DATA & MODEL  (cached)
# ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    for p in ["Salary_Data.csv", "/mnt/user-data/uploads/Salary_Data.csv"]:
        if os.path.exists(p):
            df = pd.read_csv(p)
            break
    else:
        st.error("❌  Salary_Data.csv not found. Place it in the same folder as app.py.")
        st.stop()

    # ── Columns that must be numeric ────────────────────────────────
    # On Python 3.14 / newer pandas these may be loaded as object/string
    # dtype (BOM, whitespace, stricter inference).  Coerce them first so
    # every downstream .median(), .mean(), .min(), .max() is safe.
    numeric_cols = ['Age', 'Years of Experience', 'Salary']

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df_raw = df.copy()
    df = df.drop_duplicates().reset_index(drop=True)

    # ── Fill missing values ─────────────────────────────────────────
    for col in numeric_cols:
        if col in df.columns:
            if df[col].isna().all():
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna(df[col].median())

    cat_cols = [c for c in df.columns if c not in numeric_cols]
    for col in cat_cols:
        mv = df[col].mode(dropna=True)
        if len(mv):
            df[col] = df[col].fillna(mv[0])

    df = df.dropna().reset_index(drop=True)

    gender_opts = sorted(df_raw['Gender'].dropna().unique().tolist())
    edu_opts    = sorted(df_raw['Education Level'].dropna().unique().tolist(),
                         key=lambda x: ["Bachelor's","Master's","PhD"].index(x)
                         if x in ["Bachelor's","Master's","PhD"] else 99)
    job_opts    = sorted(df_raw['Job Title'].dropna().unique().tolist())

    le_dict = {}
    for col in ['Gender', 'Education Level', 'Job Title']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le

    return df, df_raw, le_dict, gender_opts, edu_opts, job_opts


@st.cache_resource
def train_models(_df):
    X = _df.drop('Salary', axis=1)
    y = _df['Salary']
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    sc = StandardScaler()
    Xt, Xv = sc.fit_transform(Xtr), sc.transform(Xte)

    lr  = LinearRegression().fit(Xt, ytr)
    xgb = XGBRegressor(n_estimators=300, learning_rate=0.08, max_depth=5,
                        subsample=0.8, colsample_bytree=0.8,
                        random_state=42, verbosity=0).fit(Xt, ytr)

    lp, xp = lr.predict(Xv), xgb.predict(Xv)
    feat = X.columns.tolist()
    mets = {
        "Linear Regression": {"R²": r2_score(yte,lp),  "MAE": mean_absolute_error(yte,lp),
                               "RMSE": math.sqrt(mean_squared_error(yte,lp))},
        "XGBoost":           {"R²": r2_score(yte,xp),  "MAE": mean_absolute_error(yte,xp),
                               "RMSE": math.sqrt(mean_squared_error(yte,xp))}
    }
    return lr, xgb, sc, Xte.values, yte.values, lp, xp, feat, mets


df_enc, df_raw, le_dict, gender_opts, edu_opts, job_opts = load_data()
lr_m, xgb_m, scaler, X_te, y_te, lr_p, xgb_p, feat_names, metrics = train_models(df_enc)


# ─────────────────────────────────────────────────────────────────
# SIDEBAR  (PayPulse branding)
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:20px 0 6px;'>
      <div class='paypulse-float' style='position:relative;display:inline-block;'>
        <div class='paypulse-logo-pulse' style='
            width:64px;height:64px;background:linear-gradient(135deg,#4f46e5,#7c3aed);
            border-radius:18px;display:flex;align-items:center;justify-content:center;
            margin:0 auto 10px;font-size:1.8rem;'>💸</div>
      </div>
      <div style='font-family:"Space Grotesk",Inter,sans-serif;font-size:1.5rem;
                  font-weight:800;letter-spacing:-0.02em;line-height:1;'>
        <span class='paypulse-shimmer'>Pay</span><span style='color:#e2e8f0;'>Pulse</span>
      </div>
      <p style='color:#475569;font-size:0.7rem;margin:5px 0 0;letter-spacing:0.08em;
                text-transform:uppercase;'>AI Salary Intelligence</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio("", [
        "🏠  Dashboard",
        "🔮  Salary Predictor",
        "📊  Data Explorer",
        "🧠  Model Insights",
        "📦  Batch Predictor",
        "⚖️  Profile Compare",
        "📈  Career Growth",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("<p style='color:#334155;font-size:0.7rem;text-transform:uppercase;"
                "letter-spacing:1px;'>Dataset</p>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.metric("Records",    len(df_raw))
    c2.metric("Job Roles",  len(job_opts))
    c1.metric("Min Salary", f"${df_raw['Salary'].min():,.0f}")
    c2.metric("Max Salary", f"${df_raw['Salary'].max():,.0f}")

    st.markdown("---")
    st.markdown("<p style='color:#334155;font-size:0.7rem;text-transform:uppercase;"
                "letter-spacing:1px;'>Models</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:#4ade80;font-size:0.8rem;'>✅ Linear Regression "
                f"R²={metrics['Linear Regression']['R²']:.1%}</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:#4ade80;font-size:0.8rem;'>✅ XGBoost "
                f"R²={metrics['XGBoost']['R²']:.1%}</p>", unsafe_allow_html=True)

    st.markdown("""
    <div style='margin-top:24px;padding:10px;background:rgba(99,102,241,0.07);
                border-radius:10px;border:1px solid #1e2d4a;text-align:center;'>
      <p style='color:#4f46e5;font-size:0.7rem;margin:0;'>Powered by XGBoost &amp; sklearn</p>
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ═══════════════════════════════════════════════════════════════════
if "Dashboard" in page:

    # ── Animated hero banner ──────────────────────────────────────
    st.markdown(f"""
    <div class='fade-in-up' style='
        background: linear-gradient(135deg, #0d1025 0%, #1a1040 50%, #0d1825 100%);
        border: 1px solid #1e2d4a;
        border-radius: 24px;
        padding: 36px 40px 28px;
        margin-bottom: 28px;
        position: relative;
        overflow: hidden;
    '>
      <!-- background grid -->
      <div style='
        position:absolute;inset:0;
        background-image: linear-gradient(rgba(99,102,241,0.05) 1px,transparent 1px),
                          linear-gradient(90deg,rgba(99,102,241,0.05) 1px,transparent 1px);
        background-size:40px 40px;pointer-events:none;border-radius:24px;
      '></div>

      <!-- floating orbs -->
      <div style='position:absolute;top:-20px;right:60px;width:160px;height:160px;
                  background:radial-gradient(circle,rgba(124,58,237,0.18),transparent 70%);
                  border-radius:50%;pointer-events:none;'></div>
      <div style='position:absolute;bottom:-30px;left:40px;width:120px;height:120px;
                  background:radial-gradient(circle,rgba(99,102,241,0.14),transparent 70%);
                  border-radius:50%;pointer-events:none;'></div>

      <div style='position:relative;z-index:1;display:flex;align-items:center;gap:24px;flex-wrap:wrap;'>
        <div class='paypulse-float' style='
            width:80px;height:80px;
            background:linear-gradient(135deg,#4f46e5,#7c3aed);
            border-radius:22px;display:flex;align-items:center;
            justify-content:center;font-size:2.2rem;flex-shrink:0;
            box-shadow:0 0 40px rgba(124,58,237,0.4);
        '>💸</div>
        <div>
          <div style='font-family:"Space Grotesk",Inter,sans-serif;
                      font-size:2.8rem;font-weight:800;line-height:1;letter-spacing:-0.03em;'>
            <span class='paypulse-shimmer'>Pay</span><span style='color:#f1f5f9;'>Pulse</span>
          </div>
          <p style='color:#64748b;margin:6px 0 0;font-size:1rem;'>
            AI-Powered Employee Salary Intelligence Platform
          </p>
          <div style='display:flex;gap:10px;margin-top:12px;flex-wrap:wrap;'>
            <span style='background:rgba(99,102,241,0.15);color:#818cf8;border:1px solid rgba(99,102,241,0.3);
                         border-radius:99px;padding:3px 12px;font-size:0.72rem;font-weight:600;'>
              ⚡ XGBoost {metrics["XGBoost"]["R²"]:.1%} R²
            </span>
            <span style='background:rgba(34,197,94,0.1);color:#4ade80;border:1px solid rgba(34,197,94,0.25);
                         border-radius:99px;padding:3px 12px;font-size:0.72rem;font-weight:600;'>
              ✅ {len(df_raw)} Records
            </span>
            <span style='background:rgba(244,114,182,0.1);color:#f472b6;border:1px solid rgba(244,114,182,0.25);
                         border-radius:99px;padding:3px 12px;font-size:0.72rem;font-weight:600;'>
              🎯 {len(job_opts)} Job Roles
            </span>
          </div>
        </div>
      </div>

      <!-- ticker -->
      <div class='ticker-wrap' style='margin-top:22px;'>
        <div class='ticker'>
          <span>💰 Avg Salary: {fmt(df_raw["Salary"].mean())}</span>
          <span>📈 Max Salary: {fmt(df_raw["Salary"].max())}</span>
          <span>🎓 Top Edu: PhD</span>
          <span>⏱ Avg Exp: {df_raw["Years of Experience"].mean():.1f} yrs</span>
          <span>👥 Genders: {", ".join(gender_opts)}</span>
          <span>🔮 Models: Linear Regression + XGBoost</span>
          <span>💰 Avg Salary: {fmt(df_raw["Salary"].mean())}</span>
          <span>📈 Max Salary: {fmt(df_raw["Salary"].max())}</span>
          <span>🎓 Top Edu: PhD</span>
          <span>⏱ Avg Exp: {df_raw["Years of Experience"].mean():.1f} yrs</span>
          <span>👥 Genders: {", ".join(gender_opts)}</span>
          <span>🔮 Models: Linear Regression + XGBoost</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI row ────────────────────────────────────────────────────
    kpis = [
        ("Total Employees",  len(df_raw),                              "records in dataset"),
        ("Avg Salary",       f"${df_raw['Salary'].mean():,.0f}",       "across all roles"),
        ("Median Salary",    f"${df_raw['Salary'].median():,.0f}",     "50th percentile"),
        ("Max Salary",       f"${df_raw['Salary'].max():,.0f}",        "highest earner"),
        ("Avg Experience",   f"{df_raw['Years of Experience'].mean():.1f} yrs", "per employee"),
        ("Unique Job Roles", df_raw['Job Title'].nunique(),             "distinct titles"),
    ]
    cols = st.columns(6)
    for col, (label, val, sub) in zip(cols, kpis):
        col.markdown(f"""
        <div class='kpi-card fade-in-up'>
          <div class='kpi-label'>{label}</div>
          <div class='kpi-value'>{val}</div>
          <div class='kpi-sub'>{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        edu_sal = df_raw.groupby('Education Level')['Salary'].mean().reindex(edu_opts).reset_index()
        fig = go.Figure(go.Bar(
            x=edu_sal['Education Level'], y=edu_sal['Salary'],
            marker=dict(color=['#6366f1','#a78bfa','#f472b6'],
                        line=dict(color='#0d1117', width=1)),
            text=[f"${v:,.0f}" for v in edu_sal['Salary']],
            textposition='outside', textfont=dict(color='#94a3b8', size=11)
        ))
        fig.update_layout(title="Avg Salary by Education Level", yaxis_title="Salary ($)", xaxis_title="")
        st.plotly_chart(plotly_dark(fig), use_container_width=True)

    with c2:
        gen_sal = df_raw.groupby('Gender')['Salary'].mean().reset_index()
        fig = go.Figure(go.Bar(
            x=gen_sal['Gender'], y=gen_sal['Salary'],
            marker=dict(color=['#f472b6','#6366f1'],
                        line=dict(color='#0d1117', width=1)),
            text=[f"${v:,.0f}" for v in gen_sal['Salary']],
            textposition='outside', textfont=dict(color='#94a3b8', size=11)
        ))
        fig.update_layout(title="Avg Salary by Gender", yaxis_title="Salary ($)", xaxis_title="")
        st.plotly_chart(plotly_dark(fig), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        top_jobs = (df_raw.groupby('Job Title')['Salary'].mean()
                    .sort_values(ascending=False).head(10).reset_index())
        fig = go.Figure(go.Bar(
            x=top_jobs['Salary'], y=top_jobs['Job Title'], orientation='h',
            marker=dict(
                color=list(reversed(px.colors.sequential.Purples[2:][:10])),
                line=dict(color='#0d1117', width=1)
            ),
            text=[f"${v:,.0f}" for v in top_jobs['Salary']],
            textposition='outside', textfont=dict(color='#94a3b8', size=10)
        ))
        fig.update_layout(title="Top 10 Highest Paying Jobs",
                          yaxis=dict(autorange='reversed'), xaxis_title="Avg Salary ($)")
        st.plotly_chart(plotly_dark(fig, 450), use_container_width=True)

    with c4:
        fig = go.Figure(go.Histogram(
            x=df_raw['Years of Experience'], nbinsx=20,
            marker=dict(color='#6366f1', line=dict(color='#0d1117', width=1)), opacity=0.85
        ))
        fig.update_layout(title="Experience Distribution",
                          xaxis_title="Years of Experience", yaxis_title="Count")
        st.plotly_chart(plotly_dark(fig), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# PAGE 2 — SALARY PREDICTOR
# ═══════════════════════════════════════════════════════════════════
elif "Predictor" in page and "Batch" not in page:
    st.markdown("<div class='section-title'>🔮 Salary Predictor</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Fill in the employee details, then click "
                "<b>💸 Predict Salary</b> to see the AI-generated result.</div>",
                unsafe_allow_html=True)

    # Session state — only update on button click
    if "pred_result" not in st.session_state:
        st.session_state.pred_result = None

    left, mid, right = st.columns([1.2, 1.4, 1.4])

    # ── Inputs ──────────────────────────────────────────────────────
    with left:
        st.markdown("#### 📝 Employee Details")
        model_choice = st.radio("Model", ["XGBoost ⚡ (recommended)", "Linear Regression 📐"],
                                 horizontal=False)
        gender = st.selectbox("Gender",            gender_opts)
        edu    = st.selectbox("Education Level",   edu_opts)
        job    = st.selectbox("Job Title",          job_opts)
        age    = st.slider("Age",                   22, 60, 30)
        exp    = st.slider("Years of Experience",   0,  30,  5)

        if st.button("💸 Predict Salary"):
            use_xgb   = "XGBoost" in model_choice
            sel_model = xgb_m if use_xgb else lr_m
            sal       = predict(sel_model, scaler, age, gender, edu, job, exp, le_dict)
            st.session_state.pred_result = dict(
                sal=sal, model=model_choice,
                age=age, gender=gender, edu=edu, job=job, exp=exp
            )

    # ── Prediction output ────────────────────────────────────────────
    with mid:
        if st.session_state.pred_result is None:
            st.markdown("""
            <div class='placeholder-box'>
              <div class='paypulse-float' style='font-size:3rem;'>🔮</div>
              <p style='color:#334155;margin:14px 0 4px;font-size:0.95rem;'>Enter details on the left</p>
              <p style='color:#6366f1;font-weight:700;font-size:1rem;'>then click 💸 Predict Salary</p>
            </div>""", unsafe_allow_html=True)
        else:
            p   = st.session_state.pred_result
            sal = p["sal"]
            lo, hi = sal * 0.90, sal * 1.10
            model_label = "XGBoost ⚡" if "XGBoost" in p["model"] else "Linear Regression 📐"

            st.markdown(f"""
            <div class='pred-box fade-in-up'>
              <div class='pred-label'>💸 Predicted Annual Salary</div>
              <div class='pred-salary'>{fmt(sal)}</div>
              <div class='pred-range'>Confidence range: {fmt(lo)} – {fmt(hi)}</div>
              <div style='font-size:0.72rem;color:#4ade80;margin-top:10px;opacity:0.7;'>
                {model_label} &nbsp;·&nbsp; Age {p["age"]} &nbsp;·&nbsp; {p["exp"]} yrs exp
              </div>
            </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=sal,
                number=dict(prefix="$", valueformat=",.0f", font=dict(color="#22c55e", size=26)),
                delta=dict(reference=df_raw['Salary'].mean(),
                           increasing=dict(color="#22c55e"), decreasing=dict(color="#f87171"),
                           valueformat=",.0f"),
                gauge=dict(
                    axis=dict(range=[0, df_raw['Salary'].max()], tickformat="$,.0f",
                              tickfont=dict(size=9, color="#475569")),
                    bar=dict(color="#22c55e"),
                    bgcolor="#060b18", bordercolor="#1e2d4a",
                    steps=[
                        dict(range=[0, 60000],   color="#0d1525"),
                        dict(range=[60000,  120000],  color="#0d1f25"),
                        dict(range=[120000, 200000],  color="#0d2518"),
                        dict(range=[200000, df_raw['Salary'].max()], color="#0d3018"),
                    ],
                    threshold=dict(line=dict(color="#818cf8", width=3),
                                   thickness=0.75, value=df_raw['Salary'].mean())
                ),
                title=dict(text="vs. Dataset Average", font=dict(color="#64748b", size=11))
            ))
            fig.update_layout(height=240, paper_bgcolor='rgba(0,0,0,0)',
                              font=dict(color='#94a3b8'), margin=dict(t=36,b=0,l=16,r=16))
            st.plotly_chart(fig, use_container_width=True)

    # ── Tips + What-If ────────────────────────────────────────────────
    with right:
        if st.session_state.pred_result is None:
            st.markdown("""
            <div class='placeholder-box'>
              <div class='paypulse-float' style='font-size:2.5rem;'>💡</div>
              <p style='color:#334155;margin:14px 0 4px;font-size:0.9rem;'>Tips & What-If Explorer</p>
              <p style='color:#475569;font-size:0.8rem;'>appear after prediction</p>
            </div>""", unsafe_allow_html=True)
        else:
            p         = st.session_state.pred_result
            sal       = p["sal"]
            sel_model = xgb_m if "XGBoost" in p["model"] else lr_m
            ag, gn, ed, jb, ex = p["age"], p["gender"], p["edu"], p["job"], p["exp"]

            st.markdown("#### 💡 Salary Improvement Tips")
            tips = []

            if ed == "Bachelor's":
                up = predict(sel_model, scaler, ag, gn, "Master's", jb, ex, le_dict) - sal
                if up > 0: tips.append(f"🎓 Master's degree could add ~{fmt(up)} (+{up/sal:.0%})")
            if ed == "Master's":
                up = predict(sel_model, scaler, ag, gn, "PhD", jb, ex, le_dict) - sal
                if up > 0: tips.append(f"🎓 PhD could add ~{fmt(up)} (+{up/sal:.0%})")

            me = predict(sel_model, scaler, ag+3, gn, ed, jb, ex+3, le_dict) - sal
            if me > 0: tips.append(f"⏳ +3 yrs experience could add ~{fmt(me)} (+{me/sal:.0%})")

            sj = "Senior " + jb if "Senior " not in jb and ("Senior " + jb) in job_opts else None
            if sj:
                up = predict(sel_model, scaler, ag, gn, ed, sj, ex, le_dict) - sal
                if up > 0: tips.append(f"🚀 Promotion to '{sj}' could add ~{fmt(up)} (+{up/sal:.0%})")

            pct = (df_raw['Salary'] < sal).mean() * 100
            tips.append(f"📊 Your salary beats {pct:.0f}% of employees in the dataset")
            if not tips: tips.append("🌟 You're already at a top salary level!")

            for tip in tips:
                st.markdown(f"<div class='tip-card'>{tip}</div>", unsafe_allow_html=True)

            st.markdown("<br>🎛️ What-If Explorer", unsafe_allow_html=True)
            st.caption("Salary trajectory as experience increases:")

            exp_rng = list(range(0, 26))
            sal_rng = [predict(sel_model, scaler, ag, gn, ed, jb, e, le_dict) for e in exp_rng]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=exp_rng, y=sal_rng, mode='lines+markers',
                line=dict(color='#6366f1', width=2.5),
                marker=dict(size=5, color='#6366f1'),
                hovertemplate="Exp: %{x}yr → $%{y:,.0f}<extra></extra>"
            ))
            fig.add_vline(x=ex, line=dict(color='#f59e0b', width=2, dash='dash'))
            fig.add_annotation(x=ex, y=sal, text=f"You: {fmt(sal)}",
                               showarrow=True, arrowcolor='#f59e0b',
                               font=dict(color='#f59e0b', size=11), bgcolor='#111827')
            fig.update_layout(xaxis_title="Years of Experience",
                              yaxis_title="Salary ($)", showlegend=False)
            st.plotly_chart(plotly_dark(fig, 280), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# PAGE 3 — DATA EXPLORER
# ═══════════════════════════════════════════════════════════════════
elif "Data Explorer" in page:
    st.markdown("<div class='section-title'>📊 Data Explorer</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Explore the dataset with interactive charts and rotatable 3D views.</div>",
                unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🌐 3D Salary Surface", "💫 3D Scatter", "📈 Distributions",
        "🗺️ Heatmap", "🔍 Job Deep Dive"
    ])

    with tab1:
        st.markdown("##### 🌐 Salary Surface — Age × Experience")
        st.caption("Rotate, zoom and hover to explore the 3D salary landscape.")
        edu_choice_3d = st.selectbox("Education Level", edu_opts, key='surf_edu')

        age_vals = np.linspace(23, 55, 30)
        exp_vals = np.linspace(0, 25, 30)
        Z = np.zeros((len(age_vals), len(exp_vals)))
        for i, a in enumerate(age_vals):
            for j, e in enumerate(exp_vals):
                Z[i,j] = predict(xgb_m, scaler, a, "Male", edu_choice_3d,
                                 "Software Engineer", e, le_dict)

        fig = go.Figure(go.Surface(
            x=exp_vals, y=age_vals, z=Z, colorscale='Viridis',
            contours=dict(z=dict(show=True, usecolormap=True,
                                 highlightcolor='white', project_z=True)),
            hovertemplate="Exp: %{x:.0f}yr<br>Age: %{y:.0f}<br>Salary: $%{z:,.0f}<extra></extra>"
        ))
        fig.update_layout(height=560,
            scene=dict(
                xaxis=dict(title='Experience (yrs)', backgroundcolor='#060b18',
                           gridcolor='#1e2d4a', color='#94a3b8'),
                yaxis=dict(title='Age', backgroundcolor='#060b18',
                           gridcolor='#1e2d4a', color='#94a3b8'),
                zaxis=dict(title='Salary ($)', backgroundcolor='#060b18',
                           gridcolor='#1e2d4a', color='#94a3b8'),
                bgcolor='#060b18',
            ),
            paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#94a3b8'),
            margin=dict(t=32,b=0,l=0,r=0))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("##### 💫 3D Scatter — Age × Experience × Salary")
        color_by = st.radio("Color by", ["Education Level","Gender"], horizontal=True, key='s3d_col')
        fig = px.scatter_3d(df_raw.dropna(),
            x='Age', y='Years of Experience', z='Salary', color=color_by,
            opacity=0.82, color_discrete_sequence=px.colors.qualitative.Bold,
            hover_data={'Job Title': True, 'Salary': ':$,.0f'})
        fig.update_traces(marker=dict(size=4, line=dict(width=0)))
        fig.update_layout(height=560,
            scene=dict(
                xaxis=dict(title='Age', backgroundcolor='#060b18', gridcolor='#1e2d4a', color='#94a3b8'),
                yaxis=dict(title='Experience', backgroundcolor='#060b18', gridcolor='#1e2d4a', color='#94a3b8'),
                zaxis=dict(title='Salary ($)', backgroundcolor='#060b18', gridcolor='#1e2d4a', color='#94a3b8'),
                bgcolor='#060b18'),
            paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#94a3b8'),
            legend=dict(bgcolor='rgba(0,0,0,0)'), margin=dict(t=32,b=0,l=0,r=0))
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=df_raw['Salary'], nbinsx=25,
                                        marker_color='#6366f1',
                                        marker_line_color='#060b18',
                                        marker_line_width=1, opacity=0.85))
            fig.add_vline(x=df_raw['Salary'].mean(),   line=dict(color='#f59e0b', dash='dash', width=2))
            fig.add_vline(x=df_raw['Salary'].median(), line=dict(color='#a78bfa', dash='dot',  width=2))
            fig.add_annotation(x=df_raw['Salary'].mean(), y=0, yref='paper',
                               text="Mean", showarrow=False,
                               font=dict(color='#f59e0b', size=10), yshift=10)
            fig.update_layout(title="Salary Distribution",
                              xaxis_title="Salary ($)", yaxis_title="Count")
            st.plotly_chart(plotly_dark(fig), use_container_width=True)

        with c2:
            fig = px.box(df_raw, x='Education Level', y='Salary',
                         color='Education Level',
                         color_discrete_sequence=['#6366f1','#a78bfa','#f472b6'],
                         category_orders={'Education Level': edu_opts})
            fig.update_layout(title="Salary by Education (Box Plot)",
                              showlegend=False, xaxis_title="")
            st.plotly_chart(plotly_dark(fig), use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            fig = px.violin(df_raw, x='Gender', y='Salary', color='Gender',
                            box=True, points='outliers',
                            color_discrete_sequence=['#f472b6','#6366f1'])
            fig.update_layout(title="Salary by Gender (Violin)", showlegend=False)
            st.plotly_chart(plotly_dark(fig), use_container_width=True)

        with c4:
            fig = px.scatter(df_raw, x='Years of Experience', y='Salary',
                             color='Education Level', trendline='ols',
                             color_discrete_sequence=['#6366f1','#a78bfa','#f472b6'],
                             opacity=0.7, hover_data=['Job Title','Age'])
            fig.update_layout(title="Experience vs Salary (with OLS Trend)")
            st.plotly_chart(plotly_dark(fig), use_container_width=True)

    with tab4:
        st.markdown("##### 🗺️ Avg Salary Heatmap — Education × Experience Band")
        df_raw2 = df_raw.copy()
        df_raw2['Exp Band'] = pd.cut(df_raw2['Years of Experience'],
                                      bins=[0,3,7,12,18,30],
                                      labels=['0-3','4-7','8-12','13-18','19+'])
        heat = df_raw2.pivot_table(values='Salary', index='Education Level',
                                   columns='Exp Band', aggfunc='mean').reindex(edu_opts)
        fig = go.Figure(go.Heatmap(
            z=heat.values, x=heat.columns.tolist(), y=heat.index.tolist(),
            colorscale='Viridis',
            text=[[f"${v:,.0f}" if not np.isnan(v) else "N/A" for v in row]
                  for row in heat.values],
            texttemplate="%{text}", textfont=dict(size=11, color='white'),
            hovertemplate="Edu: %{y}<br>Exp: %{x}<br>Avg: %{z:$,.0f}<extra></extra>"
        ))
        fig.update_layout(title="Average Salary Heatmap",
                          xaxis_title="Experience Band (Years)", yaxis_title="")
        st.plotly_chart(plotly_dark(fig, 340), use_container_width=True)

        st.markdown("##### 🔗 Feature Correlation Matrix")
        corr = df_enc.corr()
        fig2 = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns.tolist(), y=corr.columns.tolist(),
            colorscale='RdBu', zmid=0,
            text=[[f"{v:.2f}" for v in row] for row in corr.values],
            texttemplate="%{text}", textfont=dict(size=10),
            hovertemplate="%{y} × %{x}: %{z:.2f}<extra></extra>"
        ))
        fig2.update_layout(title="Correlation Matrix")
        st.plotly_chart(plotly_dark(fig2, 380), use_container_width=True)

    with tab5:
        st.markdown("##### 🔍 Job Title Deep Dive")
        sel_jobs = st.multiselect("Select job titles to compare",
                                   job_opts, default=job_opts[:6])
        if sel_jobs:
            sub = df_raw[df_raw['Job Title'].isin(sel_jobs)]
            fig = px.box(sub, x='Job Title', y='Salary', color='Job Title',
                         color_discrete_sequence=px.colors.qualitative.Bold)
            fig.update_layout(title="Salary Range by Job Title",
                              showlegend=False, xaxis_tickangle=-30)
            st.plotly_chart(plotly_dark(fig, 480), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL INSIGHTS
# ═══════════════════════════════════════════════════════════════════
elif "Model Insights" in page:
    st.markdown("<div class='section-title'>🧠 Model Insights</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Performance metrics, feature importance, and prediction accuracy.</div>",
                unsafe_allow_html=True)

    st.markdown("### 📋 Model Report Cards")
    mc1, mc2 = st.columns(2)

    for col, (mname, color) in zip([mc1, mc2],
        [("Linear Regression","#6366f1"), ("XGBoost","#a78bfa")]):
        m = metrics[mname]
        with col:
            st.markdown(f"""
            <div style='background:#0d1525;border:1px solid {color};border-radius:16px;
                        padding:22px;box-shadow:0 0 24px {color}22;'>
              <h3 style='color:{color};margin:0 0 16px;font-size:1.05rem;font-weight:700;'>{mname}</h3>
              <div style='display:grid;grid-template-columns:1fr 1fr;gap:10px;'>
                <div style='background:#060b18;border-radius:10px;padding:14px;text-align:center;'>
                  <div style='color:#475569;font-size:0.68rem;text-transform:uppercase;
                              letter-spacing:1px;'>R² Score</div>
                  <div style='color:{color};font-size:1.7rem;font-weight:800;'>{m["R²"]:.1%}</div>
                </div>
                <div style='background:#060b18;border-radius:10px;padding:14px;text-align:center;'>
                  <div style='color:#475569;font-size:0.68rem;text-transform:uppercase;
                              letter-spacing:1px;'>MAE</div>
                  <div style='color:{color};font-size:1.7rem;font-weight:800;'>${m["MAE"]:,.0f}</div>
                </div>
                <div style='background:#060b18;border-radius:10px;padding:14px;text-align:center;'>
                  <div style='color:#475569;font-size:0.68rem;text-transform:uppercase;
                              letter-spacing:1px;'>RMSE</div>
                  <div style='color:{color};font-size:1.7rem;font-weight:800;'>${m["RMSE"]:,.0f}</div>
                </div>
                <div style='background:#060b18;border-radius:10px;padding:14px;text-align:center;'>
                  <div style='color:#475569;font-size:0.68rem;text-transform:uppercase;
                              letter-spacing:1px;'>Accuracy</div>
                  <div style='color:{color};font-size:1.7rem;font-weight:800;'>{m["R²"]:.1%}</div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### 🏆 Feature Importance (XGBoost)")
        fi_df = pd.DataFrame({'Feature': feat_names,
                              'Importance': xgb_m.feature_importances_}
                             ).sort_values('Importance', ascending=True)
        colors_fi = ['#6366f1','#a78bfa','#f472b6','#f59e0b','#4ade80']
        fig = go.Figure(go.Bar(
            x=fi_df['Importance'], y=fi_df['Feature'], orientation='h',
            marker=dict(color=colors_fi[:len(fi_df)], line=dict(color='#060b18', width=1)),
            text=[f"{v:.3f}" for v in fi_df['Importance']],
            textposition='outside', textfont=dict(color='#94a3b8', size=11)
        ))
        fig.update_layout(xaxis_title="Importance Score", yaxis_title="")
        st.plotly_chart(plotly_dark(fig, 320), use_container_width=True)

    with c2:
        st.markdown("#### 📊 LR Coefficients")
        coef_df = pd.DataFrame({'Feature': feat_names, 'Coefficient': lr_m.coef_})
        coef_df = coef_df.assign(Abs=coef_df['Coefficient'].abs()).sort_values('Abs', ascending=True)
        bar_colors = ['#f87171' if v < 0 else '#4ade80' for v in coef_df['Coefficient']]
        fig = go.Figure(go.Bar(
            x=coef_df['Coefficient'], y=coef_df['Feature'], orientation='h',
            marker=dict(color=bar_colors, line=dict(color='#060b18', width=1)),
            text=[f"{v:+.0f}" for v in coef_df['Coefficient']],
            textposition='outside', textfont=dict(color='#94a3b8', size=11)
        ))
        fig.update_layout(xaxis_title="Coefficient Value", yaxis_title="")
        st.plotly_chart(plotly_dark(fig, 320), use_container_width=True)

    st.markdown("#### 🎯 Actual vs Predicted Salary")
    m_tab1, m_tab2 = st.tabs(["XGBoost", "Linear Regression"])

    for tab, preds, col in [(m_tab1, xgb_p, '#a78bfa'), (m_tab2, lr_p, '#6366f1')]:
        with tab:
            residuals = y_te - preds
            fig = make_subplots(rows=1, cols=2,
                                subplot_titles=("Actual vs Predicted", "Residuals Distribution"))
            fig.add_trace(go.Scatter(
                x=y_te, y=preds, mode='markers',
                marker=dict(color=col, size=6, opacity=0.7,
                            line=dict(color='#060b18', width=0.5)),
                hovertemplate="Actual: $%{x:,.0f}<br>Predicted: $%{y:,.0f}<extra></extra>",
                name='Predictions'), row=1, col=1)
            mn, mx = min(y_te.min(), preds.min()), max(y_te.max(), preds.max())
            fig.add_trace(go.Scatter(x=[mn,mx], y=[mn,mx], mode='lines',
                                     line=dict(color='#f87171', dash='dash', width=2),
                                     name='Perfect Fit'), row=1, col=1)
            fig.add_trace(go.Histogram(x=residuals, nbinsx=20,
                                        marker=dict(color=col, line=dict(color='#060b18', width=1)),
                                        opacity=0.8, name='Residuals'), row=1, col=2)
            fig.add_vline(x=0, line=dict(color='#f87171', dash='dash', width=1.5), row=1, col=2)
            fig.update_layout(height=370, paper_bgcolor='rgba(0,0,0,0)',
                              plot_bgcolor='rgba(10,15,30,0.8)',
                              font=dict(color='#94a3b8', family='Inter'),
                              showlegend=False, margin=dict(t=40,b=20,l=16,r=16))
            fig.update_xaxes(gridcolor='#111827', linecolor='#1e2d4a')
            fig.update_yaxes(gridcolor='#111827', linecolor='#1e2d4a')
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 🌐 3D Prediction Comparison (LR vs XGBoost)")
    fig = go.Figure(go.Scatter3d(
        x=y_te, y=lr_p, z=xgb_p, mode='markers',
        marker=dict(size=4, color=y_te, colorscale='Viridis', opacity=0.8,
                    colorbar=dict(title='Actual Salary', thickness=12)),
        hovertemplate="Actual: $%{x:,.0f}<br>LR: $%{y:,.0f}<br>XGB: $%{z:,.0f}<extra></extra>"
    ))
    fig.update_layout(height=500,
        scene=dict(
            xaxis=dict(title='Actual Salary', backgroundcolor='#060b18',
                       gridcolor='#1e2d4a', color='#94a3b8'),
            yaxis=dict(title='LR Predicted',  backgroundcolor='#060b18',
                       gridcolor='#1e2d4a', color='#94a3b8'),
            zaxis=dict(title='XGB Predicted', backgroundcolor='#060b18',
                       gridcolor='#1e2d4a', color='#94a3b8'),
            bgcolor='#060b18'),
        paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#94a3b8', family='Inter'),
        margin=dict(t=24,b=0,l=0,r=0))
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# PAGE 5 — BATCH PREDICTOR
# ═══════════════════════════════════════════════════════════════════
elif "Batch" in page:
    st.markdown("<div class='section-title'>📦 Batch Predictor</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Upload a CSV of employees and get salary predictions for all rows at once.</div>",
                unsafe_allow_html=True)

    template_df = pd.DataFrame({
        'Age':                [30, 28, 45],
        'Gender':             ['Male', 'Female', 'Male'],
        'Education Level':    ["Bachelor's", "Master's", "PhD"],
        'Job Title':          ['Software Engineer', 'Data Analyst', 'Senior Manager'],
        'Years of Experience':[5, 3, 15]
    })
    buf = io.StringIO(); template_df.to_csv(buf, index=False)
    st.download_button("📥 Download CSV Template", buf.getvalue(),
                       "employee_template.csv", "text/csv")

    uploaded = st.file_uploader("📂 Upload Employee CSV", type=['csv'])
    if uploaded:
        in_df = pd.read_csv(uploaded)
        st.markdown("**Preview:**")
        st.dataframe(in_df.head(10), use_container_width=True)

        model_choice_b = st.radio("Model", ["XGBoost ⚡","Linear Regression 📐"], horizontal=True)
        sel_m_b = xgb_m if "XGBoost" in model_choice_b else lr_m

        if st.button("🚀 Run Batch Prediction"):
            results, errors = [], []
            for idx, row in in_df.iterrows():
                try:
                    results.append(predict(sel_m_b, scaler,
                        float(row['Age']), str(row['Gender']),
                        str(row['Education Level']), str(row['Job Title']),
                        float(row['Years of Experience']), le_dict))
                except Exception as ex:
                    results.append(None)
                    errors.append(f"Row {idx}: {ex}")

            out_df = in_df.copy()
            out_df['Predicted Salary ($)'] = results
            out_df['Salary Band'] = pd.cut(
                out_df['Predicted Salary ($)'].fillna(0),
                bins=[0,60000,100000,140000,999999],
                labels=['Entry (<$60K)','Mid ($60-100K)','Senior ($100-140K)','Executive (>$140K)']
            )
            if errors: st.warning(f"⚠️ {len(errors)} rows had errors.")

            st.markdown("### 📊 Results")
            st.dataframe(out_df.style.format({'Predicted Salary ($)': '${:,.0f}'}),
                         use_container_width=True)

            band_counts = out_df['Salary Band'].value_counts().reset_index()
            fig = go.Figure(go.Bar(
                x=band_counts['Salary Band'].astype(str),
                y=band_counts['count'],
                marker=dict(color=['#6366f1','#a78bfa','#f472b6','#f59e0b'],
                            line=dict(color='#060b18', width=1)),
                text=band_counts['count'], textposition='outside',
                textfont=dict(color='#94a3b8')
            ))
            fig.update_layout(title="Distribution of Salary Bands",
                              xaxis_title="Band", yaxis_title="Count")
            st.plotly_chart(plotly_dark(fig), use_container_width=True)

            obuf = io.StringIO(); out_df.to_csv(obuf, index=False)
            st.download_button("📥 Download Results CSV", obuf.getvalue(),
                               "predictions.csv", "text/csv")


# ═══════════════════════════════════════════════════════════════════
# PAGE 6 — PROFILE COMPARE
# ═══════════════════════════════════════════════════════════════════
elif "Compare" in page:
    st.markdown("<div class='section-title'>⚖️ Profile Comparator</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Compare salary predictions for up to 3 employee profiles side by side.</div>",
                unsafe_allow_html=True)

    num_profiles = st.radio("Profiles to compare", [2, 3], horizontal=True)
    profile_data = []
    cols = st.columns(num_profiles)

    for i, col in enumerate(cols[:num_profiles]):
        with col:
            st.markdown(f"<div class='cmp-card'><div class='cmp-name'>👤 Profile {i+1}</div>",
                        unsafe_allow_html=True)
            g  = st.selectbox("Gender",   gender_opts, key=f'g{i}')
            e  = st.selectbox("Education", edu_opts,   key=f'e{i}')
            j  = st.selectbox("Job Title",  job_opts,   key=f'j{i}',
                               index=i*10 % len(job_opts))
            ag = st.slider("Age",              22, 60, 30+i*5, key=f'a{i}')
            ex = st.slider("Experience (yrs)", 0,  30, 5+i*5,  key=f'x{i}')
            st.markdown("</div>", unsafe_allow_html=True)

            profile_data.append({
                'name': f"Profile {i+1}", 'gender': g, 'edu': e, 'job': j,
                'age': ag, 'exp': ex,
                'lr_sal':  predict(lr_m,  scaler, ag, g, e, j, ex, le_dict),
                'xgb_sal': predict(xgb_m, scaler, ag, g, e, j, ex, le_dict),
            })

    if profile_data:
        st.markdown("<br>📊 Comparison Results", unsafe_allow_html=True)

        names    = [p['name'] for p in profile_data]
        lr_vals  = [p['lr_sal']  for p in profile_data]
        xgb_vals = [p['xgb_sal'] for p in profile_data]

        fig = go.Figure(data=[
            go.Bar(name='Linear Regression', x=names, y=lr_vals,
                   marker_color='#6366f1',
                   text=[fmt(v) for v in lr_vals], textposition='outside',
                   textfont=dict(color='#94a3b8')),
            go.Bar(name='XGBoost', x=names, y=xgb_vals,
                   marker_color='#a78bfa',
                   text=[fmt(v) for v in xgb_vals], textposition='outside',
                   textfont=dict(color='#94a3b8')),
        ])
        fig.update_layout(barmode='group', title="Predicted Salary Comparison",
                          yaxis_title="Salary ($)")
        st.plotly_chart(plotly_dark(fig), use_container_width=True)

        # Radar chart — fixed fillcolor using hex_to_rgba
        categories = ['Age','Experience','Edu Level','XGB Salary','LR Salary']
        edu_map    = {"Bachelor's":1, "Master's":2, "PhD":3}
        max_vals   = [60, 30, 3, 250000, 250000]
        colors_r   = ['#6366f1','#f472b6','#4ade80']

        fig2 = go.Figure()
        for p, c in zip(profile_data, colors_r):
            vals = [
                p['age']                       / max_vals[0],
                p['exp']                       / max_vals[1],
                edu_map.get(p['edu'],1)        / max_vals[2],
                p['xgb_sal']                   / max_vals[3],
                p['lr_sal']                    / max_vals[4],
            ]
            vp = [v*100 for v in vals]
            fig2.add_trace(go.Scatterpolar(
                r=vp + [vp[0]],
                theta=categories + [categories[0]],
                fill='toself',
                name=p['name'],
                line=dict(color=c, width=2),
                fillcolor=hex_to_rgba(c, 0.12),   # ← BUG FIXED
            ))
        fig2.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0,100],
                                gridcolor='#1e2d4a', tickcolor='#334155',
                                tickfont=dict(color='#475569', size=9)),
                angularaxis=dict(gridcolor='#1e2d4a', tickcolor='#94a3b8',
                                 tickfont=dict(color='#94a3b8'))
            ),
            showlegend=True,
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#94a3b8', family='Inter'),
            height=420,
            title=dict(text="Profile Radar (normalised %)", font=dict(color='#f1f5f9'))
        )
        st.plotly_chart(fig2, use_container_width=True)

        tbl = pd.DataFrame([{
            'Profile':   p['name'], 'Job': p['job'],
            'Education': p['edu'], 'Age': p['age'],
            'Experience': f"{p['exp']} yrs",
            'LR Salary':  fmt(p['lr_sal']),
            'XGB Salary': fmt(p['xgb_sal']),
        } for p in profile_data])
        st.dataframe(tbl, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════
# PAGE 7 — CAREER GROWTH
# ═══════════════════════════════════════════════════════════════════
elif "Career" in page or "Growth" in page:
    st.markdown("<div class='section-title'>📈 Career Growth Planner</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Project salary growth over time across different career paths.</div>",
                unsafe_allow_html=True)

    c1, c2 = st.columns([1, 2])

    with c1:
        st.markdown("#### 👤 Your Profile")
        cg_gender = st.selectbox("Gender",    gender_opts, key='cg_g')
        cg_edu    = st.selectbox("Education", edu_opts,    key='cg_e')
        cg_age    = st.slider("Current Age",               22, 50, 28, key='cg_a')
        cg_exp    = st.slider("Current Experience (yrs)",  0,  20,  3, key='cg_x')
        cg_years  = st.slider("Project ahead (years)",     5,  20, 10, key='cg_y')
        cg_jobs   = st.multiselect("Compare job paths", job_opts,
                                    default=job_opts[:4] if len(job_opts)>=4 else job_opts[:2],
                                    key='cg_j')
        use_xgb_cg = "XGBoost" in st.radio("Model", ["XGBoost ⚡","Linear Regression 📐"],
                                              horizontal=False, key='cg_m')
        sel_cg_m = xgb_m if use_xgb_cg else lr_m

    with c2:
        if cg_jobs:
            fig = go.Figure()
            for idx, job in enumerate(cg_jobs):
                yr_list   = list(range(0, cg_years+1))
                sals      = [predict(sel_cg_m, scaler, cg_age+y, cg_gender,
                                     cg_edu, job, cg_exp+y, le_dict)
                             for y in yr_list]
                age_labels = [cg_age+y for y in yr_list]
                c = px.colors.qualitative.Bold[idx % len(px.colors.qualitative.Bold)]
                fig.add_trace(go.Scatter(
                    x=yr_list, y=sals, mode='lines+markers',
                    name=job[:30]+('…' if len(job)>30 else ''),
                    line=dict(color=c, width=2.5),
                    marker=dict(size=6, color=c),
                    customdata=[[age_labels[i]] for i in range(len(yr_list))],
                    hovertemplate="Year +%{x} (Age %{customdata[0]})<br>"
                                  "Salary: $%{y:,.0f}<extra>"+job+"</extra>"
                ))
            fig.update_layout(title=f"Salary Projection — Next {cg_years} Years",
                              xaxis_title="Years from Now", yaxis_title="Predicted Salary ($)")
            st.plotly_chart(plotly_dark(fig, 480), use_container_width=True)

        st.markdown("#### 🌐 3D Career Landscape")
        if cg_jobs:
            sel_job_3d = st.selectbox("Job for 3D view", cg_jobs, key='3d_job')
            age_r = np.linspace(cg_age, cg_age+cg_years, 20)
            exp_r = np.linspace(cg_exp, cg_exp+cg_years, 20)
            Z3d   = np.zeros((len(age_r), len(exp_r)))
            for i, a in enumerate(age_r):
                for j, e in enumerate(exp_r):
                    Z3d[i,j] = predict(sel_cg_m, scaler, a, cg_gender,
                                       cg_edu, sel_job_3d, e, le_dict)

            fig3d = go.Figure(go.Surface(
                x=exp_r, y=age_r, z=Z3d, colorscale='Plasma',
                contours=dict(z=dict(show=True, usecolormap=True,
                                     highlightcolor='white', project_z=False)),
                hovertemplate="Exp: %{x:.0f}yr<br>Age: %{y:.0f}<br>Salary: $%{z:,.0f}<extra></extra>"
            ))
            fig3d.update_layout(height=500,
                scene=dict(
                    xaxis=dict(title='Experience', backgroundcolor='#060b18',
                               gridcolor='#1e2d4a', color='#94a3b8'),
                    yaxis=dict(title='Age', backgroundcolor='#060b18',
                               gridcolor='#1e2d4a', color='#94a3b8'),
                    zaxis=dict(title='Salary ($)', backgroundcolor='#060b18',
                               gridcolor='#1e2d4a', color='#94a3b8'),
                    bgcolor='#060b18'),
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#94a3b8', family='Inter'),
                title=dict(text=f"Growth Landscape — {sel_job_3d}",
                           font=dict(color='#f1f5f9', size=13)),
                margin=dict(t=40,b=0,l=0,r=0))
            st.plotly_chart(fig3d, use_container_width=True)
