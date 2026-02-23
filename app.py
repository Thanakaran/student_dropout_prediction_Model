"""
Student Dropout Prediction - Streamlit Web Application
Model: XGBoost | Explainability: SHAP
"""

import os
import sys
import runpy
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Student Dropout Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: white !important;
    }

    .main {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        min-height: 100vh;
    }

    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }

    /* Force ALL text elements to white */
    p, span, div, h1, h2, h3, h4, h5, h6, li, a, label,
    .stMarkdown, .stMarkdown p, .stMarkdown span,
    .stMarkdown li, .stMarkdown a,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] span,
    [data-testid="stMarkdownContainer"] li {
        color: white !important;
    }

    /* All input labels */
    label, .stSelectbox label, .stSlider label,
    .stNumberInput label, .stTextInput label,
    .stCheckbox label, .stRadio label,
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] span {
        color: white !important;
        font-weight: 500 !important;
    }

    /* Selectbox text inside the box */
    .stSelectbox [data-baseweb="select"] div,
    .stSelectbox [data-baseweb="select"] span,
    [data-baseweb="select"] div { color: white !important; }

    /* Dropdown popup container */
    [data-baseweb="popover"],
    [data-baseweb="menu"],
    [role="listbox"],
    ul[data-baseweb="menu"] {
        background-color: #1e1b4b !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        border-radius: 8px !important;
    }

    /* Each dropdown option */
    [role="option"],
    li[role="option"],
    [data-baseweb="menu"] li,
    [data-baseweb="select"] li {
        background-color: #1e1b4b !important;
        color: white !important;
        font-size: 0.95rem !important;
    }

    /* Hover state on dropdown option */
    [role="option"]:hover,
    li[role="option"]:hover,
    [data-baseweb="menu"] li:hover {
        background-color: #4c3a8c !important;
        color: white !important;
        cursor: pointer;
    }

    /* Selected option highlight */
    [aria-selected="true"],
    [role="option"][aria-selected="true"] {
        background-color: #667eea !important;
        color: white !important;
    }


    /* Input field text (the value shown inside boxes) - dark grey */
    .stNumberInput input,
    .stTextInput input,
    input[type="number"],
    input[type="text"] {
        color: #444455 !important;
        background: rgba(255,255,255,0.92) !important;
        border: 1px solid rgba(255,255,255,0.4) !important;
        font-weight: 500;
    }

    /* Placeholder text - slightly lighter grey */
    input::placeholder { color: #888899 !important; opacity: 1 !important; }
    textarea::placeholder { color: #888899 !important; opacity: 1 !important; }
    input::-webkit-input-placeholder { color: #888899 !important; opacity: 1 !important; }
    input::-moz-placeholder { color: #888899 !important; opacity: 1 !important; }
    input:-ms-input-placeholder { color: #888899 !important; opacity: 1 !important; }


    /* Slider value label */
    .stSlider [data-testid="stTickBar"] span,
    .stSlider span { color: white !important; }

    /* Tab labels */
    .stTabs [data-baseweb="tab"] { color: white !important; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { color: white !important; font-weight: 700; }

    /* Caption / small text */
    .stCaption, .stCaption p, small { color: rgba(255,255,255,0.85) !important; }

    /* Markdown bold/italic */
    strong, em, b, i { color: white !important; }

    /* Table text */
    table, th, td { color: white !important; }

    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 40px 50px;
        margin-bottom: 30px;
        text-align: center;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.4);
    }
    .hero-banner h1 {
        color: white !important;
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }
    .hero-banner p {
        color: white !important;
        font-size: 1.1rem;
        margin-top: 10px;
    }

    /* Cards */
    .metric-card {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-4px); }
    .metric-card h3 { color: #c4b5fd !important; font-size: 0.85rem; margin: 0; text-transform: uppercase; letter-spacing: 1px; }
    .metric-card h2 { color: white !important; font-size: 2rem; font-weight: 700; margin: 5px 0 0; }

    /* Prediction Results */
    .result-dropout {
        background: linear-gradient(135deg, #ff416c, #ff4b2b);
        border-radius: 16px; padding: 30px; text-align: center;
        color: white !important; font-size: 1.8rem; font-weight: 700;
        box-shadow: 0 10px 40px rgba(255, 65, 108, 0.4);
        animation: pulse 2s infinite;
    }
    .result-graduate {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        border-radius: 16px; padding: 30px; text-align: center;
        color: white !important; font-size: 1.8rem; font-weight: 700;
        box-shadow: 0 10px 40px rgba(17, 153, 142, 0.4);
    }
    .result-enrolled {
        background: linear-gradient(135deg, #f7971e, #ffd200);
        border-radius: 16px; padding: 30px; text-align: center;
        color: white !important; font-size: 1.8rem; font-weight: 700;
        box-shadow: 0 10px 40px rgba(247, 151, 30, 0.4);
    }

    @keyframes pulse {
        0% { box-shadow: 0 10px 40px rgba(255, 65, 108, 0.4); }
        50% { box-shadow: 0 10px 60px rgba(255, 65, 108, 0.7); }
        100% { box-shadow: 0 10px 40px rgba(255, 65, 108, 0.4); }
    }

    /* Section headers */
    .section-header {
        color: #c4b5fd !important;
        font-size: 1.3rem; font-weight: 600;
        border-bottom: 2px solid rgba(196, 181, 253, 0.3);
        padding-bottom: 8px; margin-bottom: 20px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.95) !important;
        border-right: 1px solid rgba(255,255,255,0.15);
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] li,
    [data-testid="stSidebar"] label { color: white !important; }

    /* Info box */
    .info-box {
        background: rgba(102, 126, 234, 0.15);
        border: 1px solid rgba(102, 126, 234, 0.4);
        border-radius: 12px; padding: 15px 20px;
        color: white !important; margin: 10px 0;
    }
    .info-box b { color: white !important; }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white !important; border: none; border-radius: 12px;
        padding: 14px 40px; font-size: 1.1rem; font-weight: 600;
        width: 100%; cursor: pointer; transition: all 0.3s;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.6);
    }


    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# AUTO-TRAIN IF MODEL ARTIFACTS ARE MISSING
# (Runs automatically on first cloud deployment)
# ─────────────────────────────────────────────
def ensure_model_artifacts():
    """Train the model inline if pkl files are not present."""
    required = ['xgboost_model.pkl', 'label_encoder.pkl', 'feature_names.pkl', 'top_features.pkl']
    if not all(os.path.exists(f) for f in required):
        os.makedirs('plots', exist_ok=True)
        with st.spinner('First-time setup: Training the model (60-90 seconds)... Please wait.'):
            try:
                # Run train_model.py in the same Python process — no subprocess needed
                runpy.run_path('train_model.py', run_name='__main__')
            except SystemExit:
                pass
            except Exception as e:
                st.error(f'Model training failed: {e}')
                st.stop()
        # Verify artifacts were created
        if not all(os.path.exists(f) for f in required):
            st.error('Training completed but model files were not saved. Please check train_model.py.')
            st.stop()

ensure_model_artifacts()

# ─────────────────────────────────────────────
# LOAD MODEL ARTIFACTS
# ─────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model = joblib.load('xgboost_model.pkl')
    le = joblib.load('label_encoder.pkl')
    feature_names = joblib.load('feature_names.pkl')

    top_features = joblib.load('top_features.pkl')
    return model, le, feature_names, top_features

try:
    model, le, feature_names, top_features = load_artifacts()
    artifacts_loaded = True
except FileNotFoundError:
    artifacts_loaded = False

# ─────────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <h1>🎓 Student Dropout Predictor</h1>
    <p>Powered by <strong>XGBoost</strong> + <strong>SHAP Explainability</strong> · Predict academic outcomes with AI</p>
</div>
""", unsafe_allow_html=True)

if not artifacts_loaded:
    st.error("⚠️ Model artifacts not found. Please run `python train_model.py` first.")
    st.code("python train_model.py", language="bash")
    st.stop()

# ─────────────────────────────────────────────
# SIDEBAR - MODEL INFO
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 Model Info")
    st.markdown("""
    <div class="info-box">
        <b>Algorithm:</b> XGBoost<br>
        <b>Task:</b> Multi-class Classification<br>
        <b>Classes:</b> Dropout · Enrolled · Graduate<br>
        <b>Explainability:</b> SHAP
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## 📊 About the Data")
    st.markdown("""
    <div class="info-box">
        Dataset contains student academic, demographic, and socioeconomic features collected at enrollment and during the first two semesters.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## 🎯 Prediction Classes")
    st.markdown("""
    - 🔴 **Dropout** — Student left the program
    - 🟡 **Enrolled** — Student still studying
    - 🟢 **Graduate** — Student completed the program
    """)

# ─────────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📈 Model Performance", "🔍 SHAP Explainability"])

# ─────────────────────────────────────────────
# TAB 1: PREDICTION FORM
# ─────────────────────────────────────────────
with tab1:
    st.markdown('<p class="section-header">📝 Enter Student Information</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**📚 Academic (1st Semester)**")
        cu1_enrolled = st.number_input("Units Enrolled (1st Sem)", 0, 30, 6, key="cu1e")
        cu1_approved = st.number_input("Units Approved (1st Sem)", 0, 30, 5, key="cu1a")
        cu1_grade = st.number_input("Grade (1st Sem)", 0.0, 20.0, 12.5, step=0.1, key="cu1g")
        cu1_evaluations = st.number_input("Evaluations (1st Sem)", 0, 50, 8, key="cu1ev")

    with col2:
        st.markdown("**📚 Academic (2nd Semester)**")
        cu2_enrolled = st.number_input("Units Enrolled (2nd Sem)", 0, 30, 6, key="cu2e")
        cu2_approved = st.number_input("Units Approved (2nd Sem)", 0, 30, 5, key="cu2a")
        cu2_grade = st.number_input("Grade (2nd Sem)", 0.0, 20.0, 12.5, step=0.1, key="cu2g")
        cu2_evaluations = st.number_input("Evaluations (2nd Sem)", 0, 50, 8, key="cu2ev")

    with col3:
        st.markdown("**👤 Personal & Enrollment**")
        age = st.number_input("Age at Enrollment", 17, 70, 20, key="age")
        gender = st.selectbox("Gender", options=[(1, "Male"), (0, "Female")],
                              format_func=lambda x: x[1], key="gender")
        scholarship = st.selectbox("Scholarship Holder", options=[(1, "Yes"), (0, "No")],
                                   format_func=lambda x: x[1], key="scholar")
        debtor = st.selectbox("Debtor", options=[(0, "No"), (1, "Yes")],
                              format_func=lambda x: x[1], key="debtor")

    col4, col5 = st.columns(2)
    with col4:
        st.markdown("**💰 Financial & Economic**")
        tuition_uptodate = st.selectbox("Tuition Fees Up to Date", options=[(1, "Yes"), (0, "No")],
                                        format_func=lambda x: x[1], key="tuition")
        unemployment_rate = st.slider("Unemployment Rate (%)", 5.0, 20.0, 11.0, 0.1, key="unemp")
        inflation_rate = st.slider("Inflation Rate (%)", -1.0, 4.0, 1.0, 0.1, key="infl")
        gdp = st.slider("GDP Growth", -5.0, 5.0, 1.0, 0.1, key="gdp")

    with col5:
        st.markdown("**🏫 Admission Info**")
        admission_grade = st.number_input("Admission Grade", 90.0, 200.0, 130.0, step=0.5, key="adm")
        prev_qual_grade = st.number_input("Previous Qualification Grade", 90.0, 200.0, 130.0, step=0.5, key="pqg")
        displaced = st.selectbox("Displaced", options=[(0, "No"), (1, "Yes")],
                                 format_func=lambda x: x[1], key="displaced")
        international = st.selectbox("International Student", options=[(0, "No"), (1, "Yes")],
                                     format_func=lambda x: x[1], key="intl")

    st.markdown("---")

    # Build input row with all features
    def build_input_row():
        """Build a complete feature row matching the training data."""
        row = {}
        # Fill all features with median/default values first
        defaults = {
            'Marital status': 1,
            'Application mode': 1,
            'Application order': 1,
            'Course': 9500,
            'Daytime/evening attendance\t': 1,
            'Previous qualification': 1,
            'Previous qualification (grade)': prev_qual_grade,
            'Nacionality': 1,
            "Mother's qualification": 19,
            "Father's qualification": 19,
            "Mother's occupation": 5,
            "Father's occupation": 5,
            'Admission grade': admission_grade,
            'Displaced': displaced[0],
            'Educational special needs': 0,
            'Debtor': debtor[0],
            'Tuition fees up to date': tuition_uptodate[0],
            'Gender': gender[0],
            'Scholarship holder': scholarship[0],
            'Age at enrollment': age,
            'International': international[0],
            'Curricular units 1st sem (credited)': 0,
            'Curricular units 1st sem (enrolled)': cu1_enrolled,
            'Curricular units 1st sem (evaluations)': cu1_evaluations,
            'Curricular units 1st sem (approved)': cu1_approved,
            'Curricular units 1st sem (grade)': cu1_grade,
            'Curricular units 1st sem (without evaluations)': 0,
            'Curricular units 2nd sem (credited)': 0,
            'Curricular units 2nd sem (enrolled)': cu2_enrolled,
            'Curricular units 2nd sem (evaluations)': cu2_evaluations,
            'Curricular units 2nd sem (approved)': cu2_approved,
            'Curricular units 2nd sem (grade)': cu2_grade,
            'Curricular units 2nd sem (without evaluations)': 0,
            'Unemployment rate': unemployment_rate,
            'Inflation rate': inflation_rate,
            'GDP': gdp,
        }
        for feat in feature_names:
            row[feat] = defaults.get(feat, 0)
        return pd.DataFrame([row])

    predict_col, _ = st.columns([1, 2])
    with predict_col:
        predict_btn = st.button("🔮 Predict Student Outcome", use_container_width=True)

    if predict_btn:
        input_df = build_input_row()
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        predicted_class = le.inverse_transform([prediction])[0]

        st.markdown("---")
        st.markdown("### 🎯 Prediction Result")

        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            if predicted_class == 'Dropout':
                st.markdown(f'<div class="result-dropout">🔴 DROPOUT<br><small style="font-size:1rem">High Risk</small></div>', unsafe_allow_html=True)
            elif predicted_class == 'Graduate':
                st.markdown(f'<div class="result-graduate">🟢 GRADUATE<br><small style="font-size:1rem">On Track</small></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-enrolled">🟡 ENROLLED<br><small style="font-size:1rem">Currently Studying</small></div>', unsafe_allow_html=True)

        with res_col2:
            st.markdown("**Prediction Probabilities**")
            classes = le.classes_
            colors = {'Dropout': '#ff416c', 'Enrolled': '#f7971e', 'Graduate': '#11998e'}
            fig = go.Figure()
            for cls, prob in zip(classes, probabilities):
                fig.add_trace(go.Bar(
                    x=[prob * 100],
                    y=[cls],
                    orientation='h',
                    marker_color=colors.get(cls, '#667eea'),
                    text=f'{prob*100:.1f}%',
                    textposition='outside',
                    name=cls
                ))
            fig.update_layout(
                height=200,
                margin=dict(l=10, r=60, t=10, b=10),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis=dict(range=[0, 110], showgrid=False, zeroline=False),
                yaxis=dict(showgrid=False),
                showlegend=False,
                bargap=0.3
            )
            st.plotly_chart(fig, use_container_width=True)

        # SHAP Force Plot for this prediction
        st.markdown("---")
        st.markdown("### 🔍 SHAP Explanation for This Prediction")
        st.markdown("*The chart below shows which features pushed the prediction toward or away from each class.*")

        with st.spinner("Computing SHAP values..."):
            try:
                explainer = shap.TreeExplainer(model)
                shap_vals = explainer.shap_values(input_df)

                class_idx = list(le.classes_).index(predicted_class)

                if shap_vals.ndim == 3:
                    sv = shap_vals[0, :, class_idx]
                else:
                    sv = shap_vals[class_idx][0]

                feat_shap = pd.DataFrame({
                    'Feature': feature_names,
                    'SHAP Value': sv,
                    'Input Value': input_df.iloc[0].values
                }).reindex(columns=['Feature', 'SHAP Value', 'Input Value'])
                feat_shap = feat_shap.reindex(feat_shap['SHAP Value'].abs().sort_values(ascending=False).index).head(15)

                colors_shap = ['#ff416c' if v < 0 else '#11998e' for v in feat_shap['SHAP Value']]
                fig2 = go.Figure(go.Bar(
                    x=feat_shap['SHAP Value'],
                    y=feat_shap['Feature'],
                    orientation='h',
                    marker_color=colors_shap,
                    text=[f'{v:.3f}' for v in feat_shap['SHAP Value']],
                    textposition='outside'
                ))
                fig2.update_layout(
                    title=f'SHAP Values for Predicted Class: {predicted_class}',
                    height=500,
                    margin=dict(l=10, r=80, t=50, b=10),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', zeroline=True, zerolinecolor='rgba(255,255,255,0.3)'),
                    yaxis=dict(showgrid=False),
                    title_font=dict(size=14)
                )
                st.plotly_chart(fig2, use_container_width=True)
                st.caption("🟢 Green bars push toward this class | 🔴 Red bars push away from this class")
            except Exception as e:
                st.warning(f"SHAP computation skipped: {e}")

# ─────────────────────────────────────────────
# TAB 2: MODEL PERFORMANCE
# ─────────────────────────────────────────────
with tab2:
    st.markdown('<p class="section-header">📊 Model Performance Dashboard</p>', unsafe_allow_html=True)

    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown('<div class="metric-card"><h3>Accuracy</h3><h2>80.3%</h2></div>', unsafe_allow_html=True)
    with m2:
        st.markdown('<div class="metric-card"><h3>F1 (Macro)</h3><h2>0.747</h2></div>', unsafe_allow_html=True)
    with m3:
        st.markdown('<div class="metric-card"><h3>AUC (OvR)</h3><h2>0.916</h2></div>', unsafe_allow_html=True)
    with m4:
        st.markdown('<div class="metric-card"><h3>CV Accuracy</h3><h2 style="font-size:1.2rem">77.7%±0.7%</h2></div>', unsafe_allow_html=True)

    st.markdown("---")

    img_col1, img_col2 = st.columns(2)
    with img_col1:
        try:
            st.image('plots/confusion_matrix.png', caption='Confusion Matrix', use_container_width=True)
        except:
            st.info("Run `python train_model.py` to generate plots.")

    with img_col2:
        try:
            st.image('plots/feature_importance.png', caption='Feature Importance (Gain)', use_container_width=True)
        except:
            st.info("Run `python train_model.py` to generate plots.")

    st.markdown("---")
    st.markdown("### 📋 Algorithm Details")
    st.markdown("""
    | Parameter | Value |
    |-----------|-------|
    | Algorithm | XGBoost (Extreme Gradient Boosting) |
    | n_estimators | 300 |
    | max_depth | 6 |
    | learning_rate | 0.05 |
    | subsample | 0.8 |
    | colsample_bytree | 0.8 |
    | early_stopping_rounds | 20 |
    | Train/Val/Test Split | 70% / 15% / 15% |
    """)

# ─────────────────────────────────────────────
# TAB 3: SHAP EXPLAINABILITY
# ─────────────────────────────────────────────
with tab3:
    st.markdown('<p class="section-header">🔍 Global SHAP Explainability</p>', unsafe_allow_html=True)

    st.markdown("""
    **SHAP (SHapley Additive exPlanations)** assigns each feature an importance value for a particular prediction.
    It is based on game theory and provides both local (per-prediction) and global (overall model) explanations.
    """)

    st.markdown("---")

    try:
        st.image('plots/shap_global_importance.png', caption='Global SHAP Feature Importance', use_container_width=True)
    except:
        st.info("Run `python train_model.py` to generate SHAP plots.")

    st.markdown("---")
    shap_col1, shap_col2, shap_col3 = st.columns(3)
    with shap_col1:
        try:
            st.image('plots/shap_summary_dropout.png', caption='SHAP Summary - Dropout Class', use_container_width=True)
        except:
            st.info("SHAP plot not found.")
    with shap_col2:
        try:
            st.image('plots/shap_summary_enrolled.png', caption='SHAP Summary - Enrolled Class', use_container_width=True)
        except:
            st.info("SHAP plot not found.")
    with shap_col3:
        try:
            st.image('plots/shap_summary_graduate.png', caption='SHAP Summary - Graduate Class', use_container_width=True)
        except:
            st.info("SHAP plot not found.")

    st.markdown("---")
    st.markdown("""
    ### 📖 How to Read SHAP Plots
    - **Beeswarm plots**: Each dot is one student. Color = feature value (red=high, blue=low). X-axis = SHAP impact.
    - **Bar chart**: Mean absolute SHAP value — higher means more globally important.
    - **Positive SHAP** → pushes prediction toward the class.
    - **Negative SHAP** → pushes prediction away from the class.
    """)
