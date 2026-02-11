"""
Week 2 Complete Pipeline: Model Training & Selection (Days 6-10)
Run this to execute the entire Week 2 workflow
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
from sklearn.model_selection import GridSearchCV

# Set paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data' / 'final'
MODELS_DIR = BASE_DIR / 'models'
RESULTS_DIR = BASE_DIR / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

# Plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

RANDOM_STATE = 42

print("="*80)
print("WEEK 2: MODEL TRAINING & SELECTION PIPELINE")
print("="*80)
print()

# Load data
print("📂 Loading preprocessed data...")
X_train = pd.read_csv(DATA_DIR / 'X_train_scaled.csv')
y_train = pd.read_csv(DATA_DIR / 'y_train.csv').values.ravel()
X_test = pd.read_csv(DATA_DIR / 'X_test_scaled.csv')
y_test = pd.read_csv(DATA_DIR / 'y_test.csv').values.ravel()

print(f"✅ Train: {X_train.shape}, Test: {X_test.shape}")
print(f"✅ Class distribution: {(y_train==0).sum()} Good, {(y_train==1).sum()} Bad")

# ============================================================================
# DAY 6: Baseline Model
# ============================================================================

print("\n" + "="*80)
print("DAY 6: Baseline Model")
print("="*80)

print("\n🎯 Creating majority class baseline...")
# Predict majority class (Good = 0)
y_pred_baseline = np.zeros_like(y_test)

baseline_acc = accuracy_score(y_test, y_pred_baseline)
baseline_f1 = f1_score(y_test, y_pred_baseline, zero_division=0)
baseline_recall = recall_score(y_test, y_pred_baseline, zero_division=0)

baseline_results = {
    'accuracy': baseline_acc,
    'precision': 0.0,
    'recall': baseline_recall,
    'f1_score': baseline_f1,
    'roc_auc': 0.5,
    'confusion_matrix': confusion_matrix(y_test, y_pred_baseline).tolist()
}

joblib.dump(baseline_results, MODELS_DIR / 'baseline_results.pkl')
print(f"✅ Baseline F1-Score: {baseline_f1:.1%}")
print("✅ Day 6 complete!")

# ============================================================================
# DAY 7: Logistic Regression
# ============================================================================

print("\n" + "="*80)
print("DAY 7: Logistic Regression")
print("="*80)

print("\n🏋️  Training Logistic Regression...")
logreg = LogisticRegression(
    class_weight='balanced',
    random_state=RANDOM_STATE,
    max_iter=1000
)
logreg.fit(X_train, y_train)

y_pred_logreg = logreg.predict(X_test)
y_pred_proba_logreg = logreg.predict_proba(X_test)[:, 1]

logreg_acc = accuracy_score(y_test, y_pred_logreg)
logreg_prec = precision_score(y_test, y_pred_logreg, zero_division=0)
logreg_recall = recall_score(y_test, y_pred_logreg)
logreg_f1 = f1_score(y_test, y_pred_logreg, zero_division=0)
logreg_auc = roc_auc_score(y_test, y_pred_proba_logreg)

logreg_results = {
    'accuracy': logreg_acc,
    'precision': logreg_prec,
    'recall': logreg_recall,
    'f1_score': logreg_f1,
    'roc_auc': logreg_auc,
    'confusion_matrix': confusion_matrix(y_test, y_pred_logreg).tolist()
}

joblib.dump(logreg, MODELS_DIR / 'logistic_regression_model.pkl')
joblib.dump(logreg_results, MODELS_DIR / 'logistic_regression_results.pkl')
print(f"✅ LogReg F1-Score: {logreg_f1:.1%}, ROC-AUC: {logreg_auc:.3f}")
print("✅ Day 7 complete!")

# ============================================================================
# DAY 8: Random Forest
# ============================================================================

print("\n" + "="*80)
print("DAY 8: Random Forest")
print("="*80)

print("\n🏋️  Training Random Forest (with tuning)...")
rf = RandomForestClassifier(
    class_weight='balanced',
    random_state=RANDOM_STATE,
    n_jobs=-1
)

# Quick grid search
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [15, 20],
    'min_samples_split': [2, 5]
}

grid_rf = GridSearchCV(rf, param_grid_rf, cv=3, scoring='f1', n_jobs=-1, verbose=0)
grid_rf.fit(X_train, y_train)
rf_optimized = grid_rf.best_estimator_

y_pred_rf = rf_optimized.predict(X_test)
y_pred_proba_rf = rf_optimized.predict_proba(X_test)[:, 1]

rf_acc = accuracy_score(y_test, y_pred_rf)
rf_prec = precision_score(y_test, y_pred_rf, zero_division=0)
rf_recall = recall_score(y_test, y_pred_rf)
rf_f1 = f1_score(y_test, y_pred_rf, zero_division=0)
rf_auc = roc_auc_score(y_test, y_pred_proba_rf)

rf_results = {
    'accuracy': rf_acc,
    'precision': rf_prec,
    'recall': rf_recall,
    'f1_score': rf_f1,
    'roc_auc': rf_auc,
    'confusion_matrix': confusion_matrix(y_test, y_pred_rf).tolist()
}

joblib.dump(rf_optimized, MODELS_DIR / 'random_forest_model.pkl')
joblib.dump(rf_results, MODELS_DIR / 'random_forest_results.pkl')
print(f"✅ Random Forest F1-Score: {rf_f1:.1%}, ROC-AUC: {rf_auc:.3f}")
print(f"   Best params: {grid_rf.best_params_}")
print("✅ Day 8 complete!")

# ============================================================================
# DAY 9: XGBoost
# ============================================================================

print("\n" + "="*80)
print("DAY 9: XGBoost")
print("="*80)

print("\n🏋️  Training XGBoost (with tuning)...")
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

xgb_model = xgb.XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=0
)

# Quick grid search
param_grid_xgb = {
    'n_estimators': [150, 200],
    'max_depth': [6, 7],
    'learning_rate': [0.1, 0.2]
}

grid_xgb = GridSearchCV(xgb_model, param_grid_xgb, cv=3, scoring='f1', n_jobs=-1, verbose=0)
grid_xgb.fit(X_train, y_train)
xgb_optimized = grid_xgb.best_estimator_

y_pred_xgb = xgb_optimized.predict(X_test)
y_pred_proba_xgb = xgb_optimized.predict_proba(X_test)[:, 1]

xgb_acc = accuracy_score(y_test, y_pred_xgb)
xgb_prec = precision_score(y_test, y_pred_xgb, zero_division=0)
xgb_recall = recall_score(y_test, y_pred_xgb)
xgb_f1 = f1_score(y_test, y_pred_xgb, zero_division=0)
xgb_auc = roc_auc_score(y_test, y_pred_proba_xgb)

xgb_results = {
    'accuracy': xgb_acc,
    'precision': xgb_prec,
    'recall': xgb_recall,
    'f1_score': xgb_f1,
    'roc_auc': xgb_auc,
    'confusion_matrix': confusion_matrix(y_test, y_pred_xgb).tolist()
}

joblib.dump(xgb_optimized, MODELS_DIR / 'xgboost_model.pkl')
joblib.dump(xgb_results, MODELS_DIR / 'xgboost_results.pkl')
print(f"✅ XGBoost F1-Score: {xgb_f1:.1%}, ROC-AUC: {xgb_auc:.3f}")
print(f"   Best params: {grid_xgb.best_params_}")
print("✅ Day 9 complete!")

# ============================================================================
# DAY 10: Model Comparison & Selection
# ============================================================================

print("\n" + "="*80)
print("DAY 10: Model Comparison & Selection")
print("="*80)

# Create comparison table
comparison = pd.DataFrame({
    'Model': ['Baseline', 'Logistic Regression', 'Random Forest', 'XGBoost'],
    'Accuracy': [baseline_acc, logreg_acc, rf_acc, xgb_acc],
    'Precision': [0.0, logreg_prec, rf_prec, xgb_prec],
    'Recall': [baseline_recall, logreg_recall, rf_recall, xgb_recall],
    'F1-Score': [baseline_f1, logreg_f1, rf_f1, xgb_f1],
    'ROC-AUC': [0.5, logreg_auc, rf_auc, xgb_auc]
})

print("\n📊 Model Performance Comparison:")
print(comparison.to_string(index=False))

# Save comparison
comparison.to_csv(MODELS_DIR / 'model_comparison.csv', index=False)

# Select best model
best_idx = comparison['F1-Score'].idxmax()
best_model = comparison.iloc[best_idx]['Model']
best_f1 = comparison.iloc[best_idx]['F1-Score']

print(f"\n🏆 BEST MODEL: {best_model}")
print(f"   F1-Score: {best_f1:.1%}")

# Create final ROC curve comparison
print("\n📊 Creating final ROC curve...")
fig, ax = plt.subplots(figsize=(10, 8))

# Plot all models
fpr_base, tpr_base, _ = roc_curve(y_test, [0.3]*len(y_test))
fpr_log, tpr_log, _ = roc_curve(y_test, y_pred_proba_logreg)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_pred_proba_xgb)

ax.plot(fpr_base, tpr_base, 'b-', lw=2, label=f'Baseline (AUC=0.500)')
ax.plot(fpr_log, tpr_log, 'g-', lw=2, label=f'LogReg (AUC={logreg_auc:.3f})')
ax.plot(fpr_rf, tpr_rf, 'orange', lw=2, label=f'Random Forest (AUC={rf_auc:.3f})')
ax.plot(fpr_xgb, tpr_xgb, 'purple', lw=3, label=f'XGBoost (AUC={xgb_auc:.3f}) ⭐')
ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')

ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curve Comparison - All Models', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'final_roc_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Saved: {RESULTS_DIR}/final_roc_comparison.png")

print("✅ Day 10 complete!")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("🎉 WEEK 2 COMPLETE!")
print("="*80)

print(f"\n📊 Final Results:")
print(f"   Baseline:            F1={baseline_f1:.1%}, AUC=0.500")
print(f"   Logistic Regression: F1={logreg_f1:.1%}, AUC={logreg_auc:.3f}")
print(f"   Random Forest:       F1={rf_f1:.1%}, AUC={rf_auc:.3f}")
print(f"   XGBoost:             F1={xgb_f1:.1%}, AUC={xgb_auc:.3f} ⭐")

print(f"\n📁 Saved Models:")
print(f"   • {MODELS_DIR}/logistic_regression_model.pkl")
print(f"   • {MODELS_DIR}/random_forest_model.pkl")
print(f"   • {MODELS_DIR}/xgboost_model.pkl ⭐ PRODUCTION")
print(f"   • {MODELS_DIR}/model_comparison.csv")

cm_xgb = confusion_matrix(y_test, y_pred_xgb)
defaults_caught = cm_xgb[1, 1]
defaults_total = cm_xgb[1, 0] + cm_xgb[1, 1]

print(f"\n💰 Business Impact (XGBoost):")
print(f"   • Caught {defaults_caught}/{defaults_total} defaults ({defaults_caught/defaults_total*100:.1f}%)")
print(f"   • False alarms: {cm_xgb[0, 1]}/{cm_xgb[0, 0] + cm_xgb[0, 1]} ({cm_xgb[0, 1]/(cm_xgb[0, 0] + cm_xgb[0, 1])*100:.1f}%)")

print(f"\n✅ Production model ready: xgboost_model.pkl")
print(f"\nNext steps:")
print(f"  1. Review model_comparison.csv for details")
print(f"  2. Use xgboost_model.pkl for predictions")
print(f"  3. Deploy with: python src/api/main.py")
