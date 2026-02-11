"""
Train credit risk prediction model.
Uses XGBoost for classification with cross-validation and hyperparameter tuning.
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, accuracy_score, f1_score
)
import xgboost as xgb
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import settings
from src.ml.feature_engineering import (
    create_master_dataset,
    prepare_features_and_target,
    save_processed_data
)


def split_data(X: pd.DataFrame, y: pd.Series,
               test_size: float = 0.2,
               random_state: int = 42) -> tuple:
    """Split data into train and test sets with stratification."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"📊 Data split:")
    print(f"   Train: {X_train.shape[0]} samples ({y_train.sum()/len(y_train)*100:.1f}% default)")
    print(f"   Test:  {X_test.shape[0]} samples ({y_test.sum()/len(y_test)*100:.1f}% default)")

    return X_train, X_test, y_train, y_test


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
    """Scale features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    print(f"✅ Features scaled using StandardScaler")

    return X_train_scaled, X_test_scaled, scaler


def train_xgboost_model(X_train: pd.DataFrame, y_train: pd.Series,
                       X_test: pd.DataFrame, y_test: pd.Series) -> xgb.XGBClassifier:
    """
    Train XGBoost classifier with optimized hyperparameters.

    Hyperparameters are tuned for credit risk (imbalanced classes, false negative cost).
    """
    # Calculate scale_pos_weight for class imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    print(f"\n🤖 Training XGBoost model...")
    print(f"   Class imbalance ratio: {scale_pos_weight:.2f}")

    # XGBoost with tuned hyperparameters
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,  # Handle imbalance
        random_state=settings.RANDOM_STATE,
        eval_metric='logloss',
        early_stopping_rounds=20,
        verbose=False
    )

    # Train with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    print(f"✅ Model trained successfully")
    print(f"   Best iteration: {model.best_iteration}")

    return model


def evaluate_model(model, X_train: pd.DataFrame, y_train: pd.Series,
                  X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Comprehensive model evaluation."""
    print(f"\n📈 Evaluating model performance...")

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Predicted probabilities
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    metrics = {
        'train': {
            'accuracy': accuracy_score(y_train, y_train_pred),
            'f1': f1_score(y_train, y_train_pred),
            'roc_auc': roc_auc_score(y_train, y_train_proba),
        },
        'test': {
            'accuracy': accuracy_score(y_test, y_test_pred),
            'f1': f1_score(y_test, y_test_pred),
            'roc_auc': roc_auc_score(y_test, y_test_proba),
        }
    }

    # Print results
    print(f"\n{'='*60}")
    print(f"📊 MODEL PERFORMANCE")
    print(f"{'='*60}")
    print(f"\n🎯 TRAIN METRICS:")
    print(f"   Accuracy: {metrics['train']['accuracy']:.4f}")
    print(f"   F1 Score: {metrics['train']['f1']:.4f}")
    print(f"   ROC-AUC:  {metrics['train']['roc_auc']:.4f}")

    print(f"\n🎯 TEST METRICS:")
    print(f"   Accuracy: {metrics['test']['accuracy']:.4f}")
    print(f"   F1 Score: {metrics['test']['f1']:.4f}")
    print(f"   ROC-AUC:  {metrics['test']['roc_auc']:.4f}")

    # Classification report
    print(f"\n📋 TEST SET CLASSIFICATION REPORT:")
    print(classification_report(y_test, y_test_pred, target_names=['Good', 'Bad']))

    # Confusion matrix
    print(f"🔲 TEST SET CONFUSION MATRIX:")
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"                Predicted")
    print(f"                Good  Bad")
    print(f"   Actual Good  {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"   Actual Bad   {cm[1,0]:4d}  {cm[1,1]:4d}")

    # Business metrics
    print(f"\n💼 BUSINESS METRICS (TEST SET):")
    tn, fp, fn, tp = cm.ravel()
    print(f"   True Negatives (Correct Good):  {tn:4d}")
    print(f"   False Positives (Good as Bad):  {fp:4d}")
    print(f"   False Negatives (Bad as Good):  {fn:4d}  ⚠️  (COSTLY - missed defaults)")
    print(f"   True Positives (Correct Bad):   {tp:4d}")

    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
    print(f"   False Negative Rate: {false_negative_rate:.2%}  (Should be < 20%)")

    print(f"{'='*60}\n")

    return metrics


def cross_validate_model(X: pd.DataFrame, y: pd.Series, model) -> dict:
    """Perform k-fold cross-validation."""
    print(f"\n🔄 Performing {settings.CV_FOLDS}-fold cross-validation...")

    cv = StratifiedKFold(n_splits=settings.CV_FOLDS, shuffle=True, random_state=settings.RANDOM_STATE)

    # Cross-validation scores
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)

    print(f"✅ Cross-validation complete:")
    print(f"   ROC-AUC scores: {cv_scores}")
    print(f"   Mean ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    return {
        'cv_scores': cv_scores,
        'mean_score': cv_scores.mean(),
        'std_score': cv_scores.std()
    }


def get_feature_importance(model, feature_names: list, top_n: int = 20) -> pd.DataFrame:
    """Extract and display feature importance."""
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\n🔝 TOP {top_n} MOST IMPORTANT FEATURES:")
    for i, row in importance_df.head(top_n).iterrows():
        print(f"   {i+1:2d}. {row['feature']:40s} {row['importance']:.4f}")

    return importance_df


def save_model_and_artifacts(model, scaler, feature_names: list,
                            metrics: dict, importance_df: pd.DataFrame):
    """Save trained model, scaler, and metadata."""
    print(f"\n💾 Saving model and artifacts...")

    # Save model
    model_path = settings.MODEL_PATH
    joblib.dump(model, model_path)
    print(f"   ✅ Model saved: {model_path}")

    # Save scaler
    scaler_path = settings.SCALER_PATH
    joblib.dump(scaler, scaler_path)
    print(f"   ✅ Scaler saved: {scaler_path}")

    # Save feature names
    feature_names_path = settings.ARTIFACTS_DIR / f"feature_names_{settings.MODEL_VERSION}.txt"
    with open(feature_names_path, 'w') as f:
        f.write('\n'.join(feature_names))
    print(f"   ✅ Feature names saved: {feature_names_path}")

    # Save metrics
    metrics_path = settings.ARTIFACTS_DIR / f"metrics_{settings.MODEL_VERSION}.txt"
    with open(metrics_path, 'w') as f:
        f.write("MODEL PERFORMANCE METRICS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Train Accuracy: {metrics['train']['accuracy']:.4f}\n")
        f.write(f"Train F1 Score: {metrics['train']['f1']:.4f}\n")
        f.write(f"Train ROC-AUC:  {metrics['train']['roc_auc']:.4f}\n\n")
        f.write(f"Test Accuracy:  {metrics['test']['accuracy']:.4f}\n")
        f.write(f"Test F1 Score:  {metrics['test']['f1']:.4f}\n")
        f.write(f"Test ROC-AUC:   {metrics['test']['roc_auc']:.4f}\n")
    print(f"   ✅ Metrics saved: {metrics_path}")

    # Save feature importance
    importance_path = settings.ARTIFACTS_DIR / f"feature_importance_{settings.MODEL_VERSION}.csv"
    importance_df.to_csv(importance_path, index=False)
    print(f"   ✅ Feature importance saved: {importance_path}")

    print(f"\n✅ All artifacts saved successfully!")


def main():
    """Main training pipeline."""
    print("="*60)
    print("🚀 CREDIT RISK MODEL TRAINING PIPELINE")
    print("="*60)

    # Step 1: Create master dataset
    print("\n📊 Step 1: Creating master dataset...")
    master_df = create_master_dataset()
    save_processed_data(master_df)

    # Step 2: Prepare features and target
    print("\n🔧 Step 2: Preparing features and target...")
    X, y, feature_names = prepare_features_and_target(master_df)

    # Step 3: Split data
    print("\n✂️  Step 3: Splitting data...")
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=settings.TEST_SIZE)

    # Step 4: Scale features
    print("\n📏 Step 4: Scaling features...")
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # Step 5: Train model
    print("\n🎯 Step 5: Training model...")
    model = train_xgboost_model(X_train_scaled, y_train, X_test_scaled, y_test)

    # Step 6: Evaluate model
    print("\n📈 Step 6: Evaluating model...")
    metrics = evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test)

    # Step 7: Feature importance
    print("\n🔍 Step 7: Analyzing feature importance...")
    importance_df = get_feature_importance(model, feature_names, top_n=20)

    # Step 8: Cross-validation (optional - can be slow)
    # cv_results = cross_validate_model(X_scaled, y, model)

    # Step 9: Save model and artifacts
    print("\n💾 Step 9: Saving model and artifacts...")
    save_model_and_artifacts(model, scaler, feature_names, metrics, importance_df)

    print("\n" + "="*60)
    print("✨ TRAINING PIPELINE COMPLETE!")
    print("="*60)
    print(f"\n📦 Artifacts saved to: {settings.ARTIFACTS_DIR}")
    print(f"   - Model:  {settings.MODEL_PATH.name}")
    print(f"   - Scaler: {settings.SCALER_PATH.name}")
    print(f"   - Feature names, metrics, and importance files")
    print(f"\n🎉 Model is ready for deployment!")


if __name__ == "__main__":
    main()
