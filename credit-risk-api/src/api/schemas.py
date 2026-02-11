"""
Pydantic schemas for API request/response validation.
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from datetime import datetime
from enum import Enum


class RiskCategory(str, Enum):
    """Risk category enum."""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    VERY_HIGH = "Very High"


class PredictionRequest(BaseModel):
    """Request schema for single credit risk prediction."""

    # Current loan application
    loanamount: float = Field(..., gt=0, description="Loan amount requested (NGN)")
    totaldue: float = Field(..., gt=0, description="Total amount due including interest (NGN)")
    termdays: int = Field(..., gt=0, le=365*3, description="Loan term in days")

    # Demographics
    latitude_gps: float = Field(..., ge=-90, le=90, description="GPS latitude")
    longitude_gps: float = Field(..., ge=-180, le=180, description="GPS longitude")
    bank_name_clients: str = Field(..., min_length=1, description="Bank name")
    bank_branch_clients: str = Field(..., min_length=1, description="Bank branch location")
    employment_status_clients: str = Field(..., description="Employment status")
    level_of_education_clients: str = Field(..., description="Education level")

    # Historical loan features (optional - for customers with history)
    hist_num_loans: Optional[int] = Field(0, ge=0, description="Number of previous loans")
    hist_num_closed: Optional[int] = Field(0, ge=0, description="Number of closed loans")
    hist_num_open: Optional[int] = Field(0, ge=0, description="Number of open loans")
    hist_avg_loan_amount: Optional[float] = Field(0.0, ge=0, description="Average previous loan amount")
    hist_max_loan_amount: Optional[float] = Field(0.0, ge=0, description="Maximum previous loan amount")
    hist_min_loan_amount: Optional[float] = Field(0.0, ge=0, description="Minimum previous loan amount")
    hist_total_borrowed: Optional[float] = Field(0.0, ge=0, description="Total amount borrowed historically")
    hist_avg_total_due: Optional[float] = Field(0.0, ge=0, description="Average total due")
    hist_ontime_rate: Optional[float] = Field(0.0, ge=0, le=1, description="On-time payment rate")
    hist_late_rate: Optional[float] = Field(0.0, ge=0, le=1, description="Late payment rate")
    hist_never_paid_rate: Optional[float] = Field(0.0, ge=0, le=1, description="Never paid rate")
    hist_avg_days_late: Optional[float] = Field(0.0, ge=0, description="Average days late")
    hist_closure_rate: Optional[float] = Field(0.0, ge=0, le=1, description="Loan closure rate")
    days_since_last_loan: Optional[int] = Field(9999, ge=0, description="Days since last loan")
    hist_avg_term_days: Optional[float] = Field(0.0, ge=0, description="Average term days")
    hist_max_term_days: Optional[int] = Field(0, ge=0, description="Maximum term days")
    has_open_loans: Optional[int] = Field(0, ge=0, le=1, description="Has open loans (0 or 1)")
    hist_interest_burden: Optional[float] = Field(0.0, ge=0, description="Historical interest burden ratio")

    @field_validator('employment_status_clients')
    @classmethod
    def validate_employment(cls, v: str) -> str:
        valid = ['Permanent', 'Contract', 'Self-Employed', 'Temporary']
        if v not in valid:
            raise ValueError(f"employment_status_clients must be one of {valid}")
        return v

    @field_validator('level_of_education_clients')
    @classmethod
    def validate_education(cls, v: str) -> str:
        valid = ['Secondary', 'HND/BSc', 'Masters', 'PhD', 'None']
        if v not in valid:
            raise ValueError(f"level_of_education_clients must be one of {valid}")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "loanamount": 250000,
                "totaldue": 312500,
                "termdays": 180,
                "latitude_gps": 6.5244,
                "longitude_gps": 3.3792,
                "bank_name_clients": "GTBank",
                "bank_branch_clients": "Lagos",
                "employment_status_clients": "Permanent",
                "level_of_education_clients": "HND/BSc",
                "hist_num_loans": 3,
                "hist_num_closed": 3,
                "hist_num_open": 0,
                "hist_ontime_rate": 0.95,
                "days_since_last_loan": 120
            }
        }
    }


class PredictionResponse(BaseModel):
    """Response schema for credit risk prediction."""

    default_probability: float = Field(..., ge=0, le=1, description="Probability of default (0-1)")
    risk_category: RiskCategory = Field(..., description="Risk category")
    prediction: str = Field(..., description="Good or Bad")
    confidence: float = Field(..., ge=0, le=1, description="Model confidence")
    recommended_action: str = Field(..., description="Recommended lending decision")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Prediction timestamp")

    model_config = {
        "json_schema_extra": {
            "example": {
                "default_probability": 0.15,
                "risk_category": "Low",
                "prediction": "Good",
                "confidence": 0.85,
                "recommended_action": "Approve with standard terms",
                "timestamp": "2024-07-15T10:30:00Z"
            }
        }
    }


class BatchPredictionRequest(BaseModel):
    """Request schema for batch predictions."""
    requests: List[PredictionRequest] = Field(..., min_length=1, max_length=1000,
                                              description="List of prediction requests")


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""
    predictions: List[PredictionResponse]
    total_processed: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="API status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_version: str = Field(..., description="Model version")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ModelInfoResponse(BaseModel):
    """Model information response."""
    model_name: str
    model_version: str
    model_type: str = "XGBoost Classifier"
    n_features: int
    target: str = "Credit Default (Good/Bad)"
    performance_metrics: dict
    top_features: List[dict]
    last_trained: str
