# 🎓 Student Dropout Prediction

A machine learning web application that predicts whether a student will **Drop Out**, remain **Enrolled**, or **Graduate**, using **XGBoost** and **SHAP** explainability.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

---

## 📌 Features

- **Prediction** — Enter student details and get an instant prediction with probability scores
- **SHAP Explanations** — See exactly which features influenced the prediction and by how much
- **Model Performance** — View accuracy, F1, AUC, confusion matrix, and feature importance
- **Global Explainability** — SHAP beeswarm plots for each outcome class

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | **80.27%** |
| F1 Score (Macro) | **0.747** |
| AUC-ROC | **0.916** |
| 5-Fold CV | **77.7% ± 0.7%** |

---

## 🗂️ Project Structure

```
student_dropout_prediction_Model/
├── app.py                  # Streamlit web application
├── train_model.py          # XGBoost training script
├── data.csv                # Dataset (4,424 students, 37 features)
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker container config
├── docker-compose.yml      # One-command Docker deployment
└── plots/                  # Generated visualisation plots
    ├── confusion_matrix.png
    ├── feature_importance.png
    ├── shap_global_importance.png
    ├── shap_summary_dropout.png
    ├── shap_summary_enrolled.png
    └── shap_summary_graduate.png
```

---

## 🚀 Run Locally

### Option 1 — Python (recommended for development)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (first time only)
python train_model.py

# 3. Launch the app
streamlit run app.py
```

Open your browser at **http://localhost:8501**

### Option 2 — Docker (no Python setup needed)

```bash
docker-compose up --build
```

Open your browser at **http://localhost:8501**

> On first run, Docker will automatically train the model if `.pkl` files are not present.

---

## 📦 Dataset

- **Source:** [UCI ML Repository](https://doi.org/10.3390/data7110146) — Realinho et al. (2022)
- **Size:** 4,424 students · 36 features · 3 classes
- **Classes:** Dropout (32.1%), Enrolled (18.0%), Graduate (49.9%)
- **Institution:** Portuguese Higher Education Institution (2008–2019)

---

## 🤖 Algorithm

**XGBoost (Extreme Gradient Boosting)** — A sequential ensemble of decision trees that corrects errors iteratively, with built-in L1/L2 regularisation and native SHAP support.

---

## 🔍 Explainability

**SHAP (SHapley Additive exPlanations)** via `shap.TreeExplainer`:
- Global feature importance (mean |SHAP| values)
- Per-class beeswarm plots
- Per-prediction local explanation bar charts

---

## 📚 References

1. Realinho et al. (2022). *Predicting Student Dropout and Academic Success.* Data, 7(11), 146.
2. Chen & Guestrin (2016). *XGBoost: A Scalable Tree Boosting System.* KDD 2016.
3. Lundberg & Lee (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS 30.
