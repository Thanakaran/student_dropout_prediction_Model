"""
Student Dropout Prediction - Model Training Script
Algorithm: XGBoost (Extreme Gradient Boosting)
Explainability: SHAP (SHapley Additive exPlanations)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, f1_score
)
import xgboost as xgb
import shap

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("STUDENT DROPOUT PREDICTION - XGBoost Model Training")
print("=" * 60)

df = pd.read_csv('data.csv', sep=';')
print(f"\n[INFO] Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"[INFO] Target distribution:\n{df['Target'].value_counts()}\n")

# ─────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────
# Separate features and target
X = df.drop(columns=['Target'])
y = df['Target']

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"[INFO] Classes: {le.classes_} -> {list(range(len(le.classes_)))}")

# Feature names (clean up)
feature_names = X.columns.tolist()

# ─────────────────────────────────────────────
# 3. TRAIN / VALIDATION / TEST SPLIT (70/15/15)
# ─────────────────────────────────────────────
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_encoded, test_size=0.30, random_state=42, stratify=y_encoded
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"\n[INFO] Train size : {X_train.shape[0]} samples")
print(f"[INFO] Val size   : {X_val.shape[0]} samples")
print(f"[INFO] Test size  : {X_test.shape[0]} samples")

# ─────────────────────────────────────────────
# 4. XGBOOST MODEL TRAINING
# ─────────────────────────────────────────────
print("\n[INFO] Training XGBoost Classifier...")

model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=20,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

print("[INFO] Training complete!")

# ─────────────────────────────────────────────
# 5. EVALUATION
# ─────────────────────────────────────────────
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

acc = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average='macro')
f1_weighted = f1_score(y_test, y_pred, average='weighted')

# AUC (one-vs-rest for multiclass)
auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')

print("\n" + "=" * 60)
print("MODEL PERFORMANCE ON TEST SET")
print("=" * 60)
print(f"  Accuracy        : {acc:.4f} ({acc*100:.2f}%)")
print(f"  F1 (Macro)      : {f1_macro:.4f}")
print(f"  F1 (Weighted)   : {f1_weighted:.4f}")
print(f"  AUC (OvR Macro) : {auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric='mlogloss',
        random_state=42, n_jobs=-1
    ),
    X, y_encoded, cv=cv, scoring='accuracy'
)
print(f"\n5-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ─────────────────────────────────────────────
# 6. SAVE PLOTS
# ─────────────────────────────────────────────
os.makedirs('plots', exist_ok=True)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix - XGBoost', fontsize=14, fontweight='bold')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('plots/confusion_matrix.png', dpi=150)
plt.close()
print("\n[INFO] Confusion matrix saved -> plots/confusion_matrix.png")

# Feature Importance (built-in)
fig, ax = plt.subplots(figsize=(10, 8))
xgb.plot_importance(model, ax=ax, max_num_features=20, importance_type='gain',
                    title='Top 20 Features by Gain - XGBoost')
plt.tight_layout()
plt.savefig('plots/feature_importance.png', dpi=150)
plt.close()
print("[INFO] Feature importance plot saved -> plots/feature_importance.png")

# ─────────────────────────────────────────────
# 7. SHAP EXPLAINABILITY
# ─────────────────────────────────────────────
print("\n[INFO] Computing SHAP values (this may take a moment)...")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# SHAP Summary Plot (Beeswarm) - for each class
class_names = le.classes_
for i, cls in enumerate(class_names):
    plt.figure()
    shap.summary_plot(
        shap_values[:, :, i] if shap_values.ndim == 3 else shap_values[i],
        X_test,
        feature_names=feature_names,
        show=False,
        max_display=15
    )
    plt.title(f'SHAP Summary Plot - Class: {cls}', fontsize=13)
    plt.tight_layout()
    plt.savefig(f'plots/shap_summary_{cls.lower()}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] SHAP summary plot saved -> plots/shap_summary_{cls.lower()}.png")

# SHAP Bar Plot (global feature importance)
plt.figure()
if shap_values.ndim == 3:
    mean_shap = np.abs(shap_values).mean(axis=(0, 2))
else:
    mean_shap = np.abs(np.array(shap_values)).mean(axis=(0, 1))

feat_imp_df = pd.DataFrame({
    'Feature': feature_names,
    'Mean |SHAP|': mean_shap
}).sort_values('Mean |SHAP|', ascending=False).head(15)

plt.figure(figsize=(10, 7))
sns.barplot(data=feat_imp_df, x='Mean |SHAP|', y='Feature', palette='viridis')
plt.title('Global SHAP Feature Importance (Mean |SHAP value|)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/shap_global_importance.png', dpi=150)
plt.close()
print("[INFO] SHAP global importance plot saved -> plots/shap_global_importance.png")

# ─────────────────────────────────────────────
# 8. SAVE MODEL & ARTIFACTS
# ─────────────────────────────────────────────
joblib.dump(model, 'xgboost_model.pkl')
joblib.dump(le, 'label_encoder.pkl')
joblib.dump(feature_names, 'feature_names.pkl')

# Save top features for the Streamlit app
top_features = feat_imp_df['Feature'].tolist()
joblib.dump(top_features, 'top_features.pkl')

print("\n[INFO] Model saved -> xgboost_model.pkl")
print("[INFO] Label encoder saved -> label_encoder.pkl")
print("[INFO] Feature names saved -> feature_names.pkl")
print("[INFO] Top features saved -> top_features.pkl")

print("\n" + "=" * 60)
print("TRAINING COMPLETE! All artifacts saved.")
print("=" * 60)
print("\nNext steps:")
print("  1. Run: streamlit run app.py")
print("  2. Open the browser at http://localhost:8501")
