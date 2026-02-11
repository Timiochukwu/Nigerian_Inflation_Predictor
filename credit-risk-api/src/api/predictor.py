"""
Prediction service for credit risk scoring.
Handles model loading and inference.
"""
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import settings
from src.api.schemas import PredictionRequest, PredictionResponse, RiskCategory


class CreditRiskPredictor:
    """Production-grade credit risk prediction service."""

    def __init__(self):
        """Initialize predictor by loading model and artifacts."""
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.feature_importance = None
        self._load_model()

    def _load_model(self):
        """Load trained model and preprocessing artifacts."""
        try:
            # Load model
            if not settings.MODEL_PATH.exists():
                raise FileNotFoundError(f"Model not found at {settings.MODEL_PATH}")

            self.model = joblib.load(settings.MODEL_PATH)
            print(f"✅ Model loaded from {settings.MODEL_PATH}")

            # Load scaler
            if not settings.SCALER_PATH.exists():
                raise FileNotFoundError(f"Scaler not found at {settings.SCALER_PATH}")

            self.scaler = joblib.load(settings.SCALER_PATH)
            print(f"✅ Scaler loaded from {settings.SCALER_PATH}")

            # Load feature names
            feature_names_path = settings.ARTIFACTS_DIR / f"feature_names_{settings.MODEL_VERSION}.txt"
            with open(feature_names_path, 'r') as f:
                self.feature_names = [line.strip() for line in f.readlines()]
            print(f"✅ Loaded {len(self.feature_names)} feature names")

            # Load feature importance (optional)
            importance_path = settings.ARTIFACTS_DIR / f"feature_importance_{settings.MODEL_VERSION}.csv"
            if importance_path.exists():
                self.feature_importance = pd.read_csv(importance_path)

            print(f"✅ Credit Risk Predictor initialized successfully")

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise

    def _prepare_features(self, request: PredictionRequest) -> pd.DataFrame:
        """
        Convert request to feature vector matching training format.

        This method must exactly replicate the feature engineering logic from training.
        """
        # Extract request data
        data = request.model_dump()

        # Create base features (same as feature engineering)
        features = {}

        # Current loan features
        features['loanamount'] = data['loanamount']
        features['totaldue'] = data['totaldue']
        features['termdays'] = data['termdays']
        features['loan_to_due_ratio'] = data['loanamount'] / data['totaldue']
        features['interest_amount'] = data['totaldue'] - data['loanamount']
        features['implied_interest_rate'] = (data['totaldue'] / data['loanamount']) - 1

        # Loan size category
        if data['loanamount'] <= 50000:
            features['loan_size_category'] = 0
        elif data['loanamount'] <= 200000:
            features['loan_size_category'] = 1
        elif data['loanamount'] <= 500000:
            features['loan_size_category'] = 2
        else:
            features['loan_size_category'] = 3

        # Term category
        if data['termdays'] <= 60:
            features['term_category'] = 0
        elif data['termdays'] <= 180:
            features['term_category'] = 1
        else:
            features['term_category'] = 2

        # Demographics
        features['latitude_gps'] = data['latitude_gps']
        features['longitude_gps'] = data['longitude_gps']

        # Employment status (one-hot encoding)
        features['emp_Contract'] = 1 if data['employment_status_clients'] == 'Contract' else 0
        features['emp_Permanent'] = 1 if data['employment_status_clients'] == 'Permanent' else 0
        features['emp_Self-Employed'] = 1 if data['employment_status_clients'] == 'Self-Employed' else 0
        features['emp_Temporary'] = 1 if data['employment_status_clients'] == 'Temporary' else 0

        # Education level (one-hot encoding)
        features['edu_HND/BSc'] = 1 if data['level_of_education_clients'] == 'HND/BSc' else 0
        features['edu_Masters'] = 1 if data['level_of_education_clients'] == 'Masters' else 0
        features['edu_None'] = 1 if data['level_of_education_clients'] == 'None' else 0
        features['edu_PhD'] = 1 if data['level_of_education_clients'] == 'PhD' else 0
        features['edu_Secondary'] = 1 if data['level_of_education_clients'] == 'Secondary' else 0

        # Bank encoding (simple label encoding for inference)
        # In production, you'd want to save the encoder from training
        banks = ['GTBank', 'Access Bank', 'First Bank', 'Zenith Bank', 'UBA',
                 'Stanbic IBTC', 'Fidelity Bank', 'Union Bank']
        try:
            features['bank_encoded'] = banks.index(data['bank_name_clients'])
        except ValueError:
            features['bank_encoded'] = 0  # Default to first bank if unknown

        # Location encoding
        locations = ['Lagos', 'Abuja', 'Port Harcourt', 'Kano', 'Ibadan',
                     'Enugu', 'Kaduna', 'Jos', 'Benin City', 'Calabar']
        try:
            features['location_encoded'] = locations.index(data['bank_branch_clients'])
        except ValueError:
            features['location_encoded'] = 0  # Default to first location if unknown

        # Historical features (use provided values or defaults)
        features['hist_num_loans'] = data.get('hist_num_loans', 0)
        features['hist_num_closed'] = data.get('hist_num_closed', 0)
        features['hist_num_open'] = data.get('hist_num_open', 0)
        features['hist_avg_loan_amount'] = data.get('hist_avg_loan_amount', 0)
        features['hist_max_loan_amount'] = data.get('hist_max_loan_amount', 0)
        features['hist_min_loan_amount'] = data.get('hist_min_loan_amount', 0)
        features['hist_total_borrowed'] = data.get('hist_total_borrowed', 0)
        features['hist_avg_total_due'] = data.get('hist_avg_total_due', 0)
        features['hist_ontime_rate'] = data.get('hist_ontime_rate', 0)
        features['hist_late_rate'] = data.get('hist_late_rate', 0)
        features['hist_never_paid_rate'] = data.get('hist_never_paid_rate', 0)
        features['hist_avg_days_late'] = data.get('hist_avg_days_late', 0)
        features['hist_closure_rate'] = data.get('hist_closure_rate', 0)
        features['days_since_last_loan'] = data.get('days_since_last_loan', 9999)
        features['hist_avg_term_days'] = data.get('hist_avg_term_days', 0)
        features['hist_max_term_days'] = data.get('hist_max_term_days', 0)
        features['has_open_loans'] = data.get('has_open_loans', 0)
        features['hist_interest_burden'] = data.get('hist_interest_burden', 0)

        # Create DataFrame with correct feature order
        df = pd.DataFrame([features])

        # Ensure all expected features are present in correct order
        for feat in self.feature_names:
            if feat not in df.columns:
                df[feat] = 0

        # Reorder columns to match training
        df = df[self.feature_names]

        return df

    def _categorize_risk(self, probability: float) -> Tuple[RiskCategory, str]:
        """
        Categorize risk based on default probability.

        Returns:
            Tuple of (risk_category, recommended_action)
        """
        if probability < 0.15:
            return RiskCategory.LOW, "Approve with standard terms"
        elif probability < 0.30:
            return RiskCategory.MEDIUM, "Approve with careful monitoring"
        elif probability < 0.50:
            return RiskCategory.HIGH, "Approve with higher interest rate or collateral"
        else:
            return RiskCategory.VERY_HIGH, "Reject application"

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """
        Make single credit risk prediction.

        Args:
            request: PredictionRequest with customer and loan data

        Returns:
            PredictionResponse with prediction and risk assessment
        """
        try:
            # Prepare features
            X = self._prepare_features(request)

            # Scale features
            X_scaled = self.scaler.transform(X)

            # Predict
            prediction_proba = self.model.predict_proba(X_scaled)[0]
            default_probability = float(prediction_proba[1])  # Probability of Bad (class 1)
            good_probability = float(prediction_proba[0])  # Probability of Good (class 0)

            # Determine prediction and confidence
            prediction = "Bad" if default_probability > 0.5 else "Good"
            confidence = max(default_probability, good_probability)

            # Categorize risk
            risk_category, recommended_action = self._categorize_risk(default_probability)

            # Create response
            response = PredictionResponse(
                default_probability=default_probability,
                risk_category=risk_category,
                prediction=prediction,
                confidence=confidence,
                recommended_action=recommended_action
            )

            return response

        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")

    def predict_batch(self, requests: List[PredictionRequest]) -> List[PredictionResponse]:
        """
        Make batch predictions.

        Args:
            requests: List of PredictionRequest objects

        Returns:
            List of PredictionResponse objects
        """
        return [self.predict(req) for req in requests]

    def get_model_info(self) -> Dict:
        """Get model information and metadata."""
        metrics_path = settings.ARTIFACTS_DIR / f"metrics_{settings.MODEL_VERSION}.txt"
        metrics_text = ""
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                metrics_text = f.read()

        # Parse metrics (simple extraction)
        metrics = {}
        for line in metrics_text.split('\n'):
            if 'Test' in line and ':' in line:
                key, value = line.split(':')
                key = key.strip().replace('Test ', '')
                try:
                    metrics[key] = float(value.strip())
                except:
                    pass

        # Top features
        top_features = []
        if self.feature_importance is not None:
            for _, row in self.feature_importance.head(10).iterrows():
                top_features.append({
                    "feature": row['feature'],
                    "importance": float(row['importance'])
                })

        return {
            "model_name": settings.MODEL_NAME,
            "model_version": settings.MODEL_VERSION,
            "n_features": len(self.feature_names),
            "performance_metrics": metrics,
            "top_features": top_features
        }


# Global predictor instance (singleton)
predictor: CreditRiskPredictor = None


def get_predictor() -> CreditRiskPredictor:
    """Get or create predictor instance (dependency injection for FastAPI)."""
    global predictor
    if predictor is None:
        predictor = CreditRiskPredictor()
    return predictor
