# DAY 12: Advanced Request Validation & Error Handling

**Goal**: Add robust input validation, custom error messages, and comprehensive error handling to your API

**Time**: 2-3 hours

**Prerequisites**: Completed Day 11 (basic FastAPI app running)

---

## 📋 What You'll Build Today

By the end of Day 12, you'll have:
- ✅ Custom Pydantic validators for business logic validation
- ✅ Comprehensive input sanitization and data cleaning
- ✅ Global exception handlers for consistent error responses
- ✅ Detailed error messages that help API users
- ✅ Request/response logging for debugging
- ✅ Input validation for edge cases (negative values, missing fields, etc.)

---

## PART 1: Advanced Pydantic Validators

### Step 1: Update `src/api/schemas/prediction.py` with Validators

Replace your previous `LoanApplication` schema with this enhanced version:

```python
# src/api/schemas/prediction.py
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Literal, Optional
import re

class LoanApplication(BaseModel):
    """Input schema for credit risk prediction with validation"""

    # Demographics
    age: int = Field(..., ge=18, le=100, description="Applicant's age (18-100)")
    employment_status: str = Field(..., description="Employment status")
    education_level: str = Field(..., description="Highest education level")

    # Current Loan Details
    loan_amount: float = Field(..., gt=0, description="Requested loan amount (must be positive)")
    loan_term_months: int = Field(..., ge=1, le=360, description="Loan term in months (1-360)")
    interest_rate: float = Field(..., ge=0, le=100, description="Annual interest rate (0-100%)")

    # Historical Features
    total_previous_loans: int = Field(default=0, ge=0, description="Number of previous loans")
    avg_repayment_ratio: float = Field(default=1.0, ge=0, le=2, description="Average repayment ratio")
    default_history_count: int = Field(default=0, ge=0, description="Number of past defaults")

    # Optional fields
    monthly_income: Optional[float] = Field(None, gt=0, description="Monthly income (optional)")

    # CUSTOM VALIDATORS

    @field_validator('employment_status')
    @classmethod
    def validate_employment_status(cls, v: str) -> str:
        """Validate employment status is one of allowed values"""
        allowed_statuses = [
            'Permanent', 'Contract', 'Self-Employed',
            'Temporary', 'Unemployed', 'Retired', 'Student'
        ]

        # Normalize: strip whitespace, title case
        v = v.strip().title()

        if v not in allowed_statuses:
            raise ValueError(
                f"Invalid employment_status '{v}'. "
                f"Must be one of: {', '.join(allowed_statuses)}"
            )
        return v

    @field_validator('education_level')
    @classmethod
    def validate_education_level(cls, v: str) -> str:
        """Validate education level is one of allowed values"""
        allowed_levels = [
            'Primary', 'Secondary', 'High School',
            'Diploma', 'Bachelor', 'Master', 'PhD', 'None'
        ]

        v = v.strip().title()

        if v not in allowed_levels:
            raise ValueError(
                f"Invalid education_level '{v}'. "
                f"Must be one of: {', '.join(allowed_levels)}"
            )
        return v

    @field_validator('loan_amount')
    @classmethod
    def validate_loan_amount(cls, v: float) -> float:
        """Validate loan amount is within reasonable bounds"""
        MIN_LOAN = 1000
        MAX_LOAN = 10_000_000

        if v < MIN_LOAN:
            raise ValueError(f"Loan amount must be at least {MIN_LOAN}")

        if v > MAX_LOAN:
            raise ValueError(f"Loan amount cannot exceed {MAX_LOAN}")

        return round(v, 2)  # Round to 2 decimal places

    @field_validator('age')
    @classmethod
    def validate_age(cls, v: int) -> int:
        """Additional age validation"""
        if v < 18:
            raise ValueError("Applicant must be at least 18 years old")

        if v > 100:
            raise ValueError("Invalid age: must be 100 or less")

        return v

    @field_validator('default_history_count')
    @classmethod
    def validate_defaults(cls, v: int) -> int:
        """Validate default history"""
        if v > 10:
            raise ValueError(
                f"Too many defaults ({v}). Maximum allowed is 10. "
                "Please verify this data."
            )
        return v

    @model_validator(mode='after')
    def validate_debt_to_income_ratio(self):
        """Cross-field validation: check debt-to-income ratio if income provided"""
        if self.monthly_income:
            # Estimate monthly payment (simple amortization)
            monthly_rate = (self.interest_rate / 100) / 12
            num_payments = self.loan_term_months

            if monthly_rate > 0:
                monthly_payment = (
                    self.loan_amount * monthly_rate *
                    (1 + monthly_rate) ** num_payments
                ) / ((1 + monthly_rate) ** num_payments - 1)
            else:
                monthly_payment = self.loan_amount / num_payments

            debt_to_income = monthly_payment / self.monthly_income

            if debt_to_income > 0.5:
                raise ValueError(
                    f"Debt-to-income ratio too high ({debt_to_income:.2%}). "
                    "Monthly payment would be more than 50% of income."
                )

        return self

    class Config:
        json_schema_extra = {
            "example": {
                "age": 35,
                "employment_status": "Permanent",
                "education_level": "Bachelor",
                "loan_amount": 50000.0,
                "loan_term_months": 24,
                "interest_rate": 12.5,
                "total_previous_loans": 2,
                "avg_repayment_ratio": 1.1,
                "default_history_count": 0,
                "monthly_income": 5000.0
            }
        }


class ValidationError(BaseModel):
    """Detailed validation error response"""
    field: str
    message: str
    invalid_value: Optional[str] = None


class ErrorResponse(BaseModel):
    """Standard error response format"""
    error: str
    detail: str
    errors: Optional[list[ValidationError]] = None


class PredictionResponse(BaseModel):
    """Output schema for credit risk prediction"""
    prediction: Literal["Good", "Bad"]
    probability_good: float = Field(..., ge=0, le=1)
    probability_bad: float = Field(..., ge=0, le=1)
    risk_score: int = Field(..., ge=0, le=100)
    recommendation: str
    confidence_level: str  # NEW: Low, Medium, High

    class Config:
        json_schema_extra = {
            "example": {
                "prediction": "Good",
                "probability_good": 0.85,
                "probability_bad": 0.15,
                "risk_score": 15,
                "recommendation": "APPROVE - Low risk applicant",
                "confidence_level": "High"
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    api_version: str
```

**💡 What's New:**
- **Field validators:** Check individual field values
- **Model validators:** Cross-field validation (e.g., debt-to-income ratio)
- **Data normalization:** Automatically clean/format inputs
- **Business logic:** Realistic constraints (loan limits, age limits)
- **Detailed errors:** Tell users exactly what's wrong

---

## PART 2: Global Exception Handlers

### Step 2: Create `src/api/exceptions.py`

Create custom exception handlers for consistent error responses:

```python
# src/api/exceptions.py
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError as PydanticValidationError
import logging

logger = logging.getLogger(__name__)


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError
) -> JSONResponse:
    """
    Handle Pydantic validation errors with detailed messages
    """
    errors = []

    for error in exc.errors():
        field = " -> ".join(str(loc) for loc in error['loc'])
        errors.append({
            "field": field,
            "message": error['msg'],
            "invalid_value": error.get('input')
        })

    logger.warning(f"Validation error on {request.url.path}: {errors}")

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation Error",
            "detail": "One or more fields failed validation",
            "errors": errors
        }
    )


async def general_exception_handler(
    request: Request,
    exc: Exception
) -> JSONResponse:
    """
    Catch-all handler for unexpected errors
    """
    logger.error(f"Unexpected error on {request.url.path}: {str(exc)}", exc_info=True)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred. Please try again later.",
            "request_id": id(request)  # For debugging
        }
    )


class PredictionError(Exception):
    """Custom exception for prediction errors"""
    pass


async def prediction_exception_handler(
    request: Request,
    exc: PredictionError
) -> JSONResponse:
    """
    Handle prediction-specific errors
    """
    logger.error(f"Prediction error: {str(exc)}")

    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "Prediction Error",
            "detail": str(exc)
        }
    )
```

---

## PART 3: Update Main App with Exception Handlers

### Step 3: Update `src/api/main.py`

Add the exception handlers to your FastAPI app:

```python
# src/api/main.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager
import logging
import time

from .config import settings
from .routers import predictions
from .schemas.prediction import HealthResponse
from .services import model_service as ms
from .exceptions import (
    validation_exception_handler,
    general_exception_handler,
    prediction_exception_handler,
    PredictionError
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    logger.info("🚀 Starting Credit Risk API...")
    ms.model_service = ms.ModelService(settings.model_path)
    logger.info("✅ Model loaded successfully")
    yield
    logger.info("👋 Shutting down API...")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=settings.app_description,
    lifespan=lifespan
)

# Add exception handlers
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(PredictionError, prediction_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests and response times"""
    start_time = time.time()

    # Log request
    logger.info(f"→ {request.method} {request.url.path}")

    # Process request
    response = await call_next(request)

    # Calculate duration
    duration = time.time() - start_time

    # Log response
    logger.info(
        f"← {request.method} {request.url.path} "
        f"[{response.status_code}] {duration:.3f}s"
    )

    return response


# Include routers
app.include_router(predictions.router)


@app.get("/", tags=["root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to Credit Risk Scoring API",
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["monitoring"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=ms.model_service.is_loaded() if ms.model_service else False,
        api_version=settings.app_version
    )
```

---

## PART 4: Update Prediction Endpoint with Better Error Handling

### Step 4: Update `src/api/routers/predictions.py`

```python
# src/api/routers/predictions.py
from fastapi import APIRouter, HTTPException, Depends
from ..schemas.prediction import LoanApplication, PredictionResponse
from ..services.model_service import get_model_service, ModelService
from ..exceptions import PredictionError
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["predictions"])


@router.post("/predict", response_model=PredictionResponse)
async def predict_credit_risk(
    application: LoanApplication,
    model_service: ModelService = Depends(get_model_service)
):
    """
    Predict credit risk for a loan application

    **Validation Rules:**
    - Age: 18-100 years
    - Loan amount: 1,000 - 10,000,000
    - Employment status: Permanent, Contract, Self-Employed, etc.
    - Debt-to-income ratio: < 50% (if monthly_income provided)

    **Returns:**
    - Prediction: Good or Bad
    - Probabilities: Individual probabilities for each class
    - Risk score: 0-100 (higher = riskier)
    - Recommendation: APPROVE, MANUAL REVIEW, or REJECT
    - Confidence level: High, Medium, or Low
    """
    try:
        # Convert Pydantic model to dict
        features = application.model_dump()

        logger.info(f"Processing prediction for loan_amount={features['loan_amount']}")

        # Make prediction
        result = model_service.predict(features)

        logger.info(
            f"Prediction: {result['prediction']} "
            f"(risk: {result['risk_score']}, "
            f"confidence: {result.get('confidence_level', 'N/A')})"
        )

        return PredictionResponse(**result)

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise PredictionError(str(e))

    except Exception as e:
        logger.error(f"Unexpected error during prediction: {str(e)}", exc_info=True)
        raise PredictionError("Failed to generate prediction. Please check your input data.")
```

---

## PART 5: Enhanced Model Service with Confidence Scores

### Step 5: Update `src/api/services/model_service.py`

Add confidence level calculation:

```python
# src/api/services/model_service.py
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class ModelService:
    """Service for loading ML model and making predictions"""

    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.model = None
        self.feature_names = None
        self._load_model()

    def _load_model(self):
        """Load the trained model from disk"""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model not found at {self.model_path}")

            self.model = joblib.load(self.model_path)

            if hasattr(self.model, 'feature_names_in_'):
                self.feature_names = self.model.feature_names_in_

            logger.info(f"✅ Model loaded successfully from {self.model_path}")

        except Exception as e:
            logger.error(f"❌ Failed to load model: {str(e)}")
            raise

    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None

    def _calculate_confidence(self, prob_good: float, prob_bad: float) -> str:
        """
        Calculate confidence level based on probability spread

        High confidence: one probability > 0.8
        Medium confidence: one probability 0.6-0.8
        Low confidence: probabilities close (both 0.4-0.6)
        """
        max_prob = max(prob_good, prob_bad)

        if max_prob >= 0.8:
            return "High"
        elif max_prob >= 0.6:
            return "Medium"
        else:
            return "Low"

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a prediction from input features

        Args:
            features: Dictionary of feature values

        Returns:
            Dictionary with prediction, probabilities, risk score, confidence
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")

        try:
            # Convert to DataFrame
            df = pd.DataFrame([features])

            # Validate feature count (optional but recommended)
            if self.feature_names is not None:
                missing_features = set(self.feature_names) - set(df.columns)
                if missing_features:
                    raise ValueError(f"Missing required features: {missing_features}")

            # Make prediction
            prediction = self.model.predict(df)[0]
            probabilities = self.model.predict_proba(df)[0]

            prob_good = float(probabilities[0])
            prob_bad = float(probabilities[1])

            # Risk score (0-100)
            risk_score = int(prob_bad * 100)

            # Confidence level
            confidence = self._calculate_confidence(prob_good, prob_bad)

            # Recommendation with confidence consideration
            if risk_score < 30:
                recommendation = "APPROVE - Low risk applicant"
            elif risk_score < 60:
                if confidence == "Low":
                    recommendation = "MANUAL REVIEW - Moderate risk (low confidence)"
                else:
                    recommendation = "MANUAL REVIEW - Moderate risk"
            else:
                if confidence == "Low":
                    recommendation = "MANUAL REVIEW - High risk but uncertain"
                else:
                    recommendation = "REJECT - High risk applicant"

            return {
                "prediction": "Good" if prediction == 0 else "Bad",
                "probability_good": prob_good,
                "probability_bad": prob_bad,
                "risk_score": risk_score,
                "recommendation": recommendation,
                "confidence_level": confidence
            }

        except ValueError as e:
            # Re-raise validation errors
            raise

        except Exception as e:
            logger.error(f"Prediction error: {str(e)}", exc_info=True)
            raise ValueError(f"Prediction failed: {str(e)}")


# Global model service instance
model_service = None

def get_model_service() -> ModelService:
    """Dependency injection for model service"""
    if model_service is None:
        raise RuntimeError("Model service not initialized")
    return model_service
```

---

## PART 6: Test Your Enhanced Validation

### Step 6: Restart Your API

```bash
uvicorn src.api.main:app --reload
```

### Step 7: Test Invalid Inputs

Try these **invalid** requests to see error handling in action:

#### Test 1: Invalid Employment Status
```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "employment_status": "InvalidStatus",
    "education_level": "Bachelor",
    "loan_amount": 50000,
    "loan_term_months": 24,
    "interest_rate": 12.5
  }'
```

**Expected Error:**
```json
{
  "error": "Validation Error",
  "detail": "One or more fields failed validation",
  "errors": [
    {
      "field": "body -> employment_status",
      "message": "Invalid employment_status 'Invalidstatus'. Must be one of: Permanent, Contract, Self-Employed, Temporary, Unemployed, Retired, Student",
      "invalid_value": "InvalidStatus"
    }
  ]
}
```

#### Test 2: Negative Loan Amount
```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "employment_status": "Permanent",
    "education_level": "Bachelor",
    "loan_amount": -5000,
    "loan_term_months": 24,
    "interest_rate": 12.5
  }'
```

#### Test 3: High Debt-to-Income Ratio
```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "employment_status": "Permanent",
    "education_level": "Bachelor",
    "loan_amount": 50000,
    "loan_term_months": 24,
    "interest_rate": 12.5,
    "monthly_income": 1000
  }'
```

**Expected Error:**
```json
{
  "error": "Validation Error",
  "detail": "One or more fields failed validation",
  "errors": [
    {
      "field": "body",
      "message": "Debt-to-income ratio too high (234.85%). Monthly payment would be more than 50% of income."
    }
  ]
}
```

---

## ✅ Day 12 Checklist

- [ ] Custom field validators added (employment, education, loan amount)
- [ ] Model validator for debt-to-income ratio working
- [ ] Global exception handlers implemented
- [ ] Request/response logging middleware active
- [ ] Confidence level added to predictions
- [ ] Detailed error messages return helpful info
- [ ] Tested with invalid inputs (all return proper errors)
- [ ] API still works with valid inputs

---

## 🎯 What You Learned Today

1. **Advanced Pydantic Validation**
   - Field validators for individual fields
   - Model validators for cross-field validation
   - Custom error messages

2. **Error Handling Best Practices**
   - Global exception handlers
   - Consistent error response format
   - Logging for debugging

3. **Request/Response Middleware**
   - Automatic request logging
   - Response time tracking
   - Performance monitoring

4. **Production-Ready API Design**
   - Input sanitization
   - Business logic validation
   - Confidence scoring

---

## 🚀 Next Steps

**Tomorrow (Day 13):** Add authentication (JWT tokens), API key validation, and rate limiting to secure your API.

---

## 💡 Troubleshooting

**Problem:** Validation passes when it shouldn't
- **Solution:** Check that validator function returns the value (not None)

**Problem:** Too many validation errors
- **Solution:** Adjust validation rules to match your actual data distribution

**Problem:** Model predictions fail after adding validation
- **Solution:** Ensure your feature names in `LoanApplication` match your trained model exactly

---

**🎉 Your API is now robust and production-ready with proper validation! 🎉**
