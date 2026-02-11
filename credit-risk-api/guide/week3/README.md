# 🟡 Week 3: FastAPI Development

**Build a production REST API for your ML model**

---

## Overview

Transform your trained model into a **production-grade REST API** that can serve predictions via HTTP requests.

### What You'll Build

- ✅ `src/api/main.py` - FastAPI application
- ✅ `src/api/routes.py` - API endpoints
- ✅ `src/api/predictor.py` - ML inference service
- ✅ `src/api/schemas.py` - Request/response validation
- ✅ Interactive API documentation (Swagger UI)

### Daily Breakdown

| Day | Focus | Time | Deliverable |
|-----|-------|------|-------------|
| **Day 11** | FastAPI Setup & Schemas | 1h | Working API skeleton |
| **Day 12** | Prediction Service | 1h | Model loading & inference |
| **Day 13** | API Routes | 45m | All endpoints implemented |
| **Day 14** | Testing | 45m | Test suite passing |
| **Day 15** | Documentation | 30m | Complete API docs |

**Total Time:** ~4 hours

---

## Learning Outcomes

1. **REST API Design**
   - HTTP methods (GET, POST)
   - Request/response patterns
   - Error handling

2. **FastAPI Framework**
   - Dependency injection
   - Automatic validation
   - Auto-generated documentation

3. **ML Model Serving**
   - Loading saved models
   - Feature preprocessing
   - Real-time inference

4. **Production Practices**
   - Health checks
   - Logging
   - Error handling

---

## Prerequisites

- ✅ Week 2 complete (trained XGBoost model)
- ✅ Model artifacts in `models/artifacts/`
- ✅ FastAPI installed (`pip install fastapi uvicorn`)

---

## Quick Start

**Option A:** Use pre-built API:

```bash
cd credit-risk-api

# Start API server
uvicorn src.api.main:app --reload

# API runs at: http://localhost:8000
# Docs at: http://localhost:8000/docs
```

**Option B:** Build step-by-step following daily guides

---

## API Endpoints

### 1. Root Endpoint
```bash
GET /
```
Returns API information and available endpoints

### 2. Health Check
```bash
GET /api/v1/health
```
Check if API and model are ready

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "v1"
}
```

### 3. Single Prediction
```bash
POST /api/v1/predict
```
Predict credit risk for one application

**Request:**
```json
{
  "loanamount": 250000,
  "totaldue": 312500,
  "termdays": 180,
  "employment_status_clients": "Permanent",
  "level_of_education_clients": "HND/BSc",
  "hist_num_loans": 3,
  "hist_ontime_rate": 0.95
}
```

**Response:**
```json
{
  "default_probability": 0.15,
  "risk_category": "Low",
  "prediction": "Good",
  "confidence": 0.85,
  "recommended_action": "Approve with standard terms"
}
```

### 4. Batch Prediction
```bash
POST /api/v1/predict/batch
```
Process multiple applications at once

### 5. Model Info
```bash
GET /api/v1/model/info
```
Get model metadata and performance metrics

---

## Day-by-Day Guide

### Day 11: FastAPI Setup & Schemas

**Create `src/api/schemas.py`:**
```python
from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    loanamount: float = Field(..., gt=0)
    totaldue: float = Field(..., gt=0)
    termdays: int = Field(..., gt=0, le=365*3)
    # ... all required fields

class PredictionResponse(BaseModel):
    default_probability: float
    risk_category: str
    prediction: str
    confidence: float
    recommended_action: str
```

**Create `src/api/main.py`:**
```python
from fastapi import FastAPI

app = FastAPI(
    title="Credit Risk Scoring API",
    version="1.0.0"
)

@app.get("/")
def root():
    return {"message": "Credit Risk API"}

@app.get("/api/v1/health")
def health():
    return {"status": "healthy"}
```

**Run it:**
```bash
uvicorn src.api.main:app --reload
```

Visit: http://localhost:8000/docs for interactive docs!

---

### Day 12: Prediction Service

**Create `src/api/predictor.py`:**

This loads your trained model and handles inference:

```python
import joblib
import pandas as pd
from pathlib import Path

class CreditRiskPredictor:
    def __init__(self):
        self.model = joblib.load('models/artifacts/credit_risk_xgboost_v1.pkl')
        self.scaler = joblib.load('models/artifacts/scaler_v1.pkl')

    def predict(self, request):
        # Convert request to features
        features = self._prepare_features(request)

        # Scale
        features_scaled = self.scaler.transform(features)

        # Predict
        proba = self.model.predict_proba(features_scaled)[0][1]

        # Categorize risk
        risk_category = self._categorize_risk(proba)

        return {
            "default_probability": proba,
            "risk_category": risk_category,
            # ...
        }
```

---

### Day 13: API Routes

Connect everything in `src/api/routes.py`:

```python
from fastapi import APIRouter
from .predictor import CreditRiskPredictor
from .schemas import PredictionRequest, PredictionResponse

router = APIRouter()
predictor = CreditRiskPredictor()

@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    return predictor.predict(request)
```

Add routes to main app:
```python
from .routes import router
app.include_router(router, prefix="/api/v1")
```

---

### Day 14: Testing

**Create `test_api.py`:**

```python
import requests

def test_health():
    response = requests.get("http://localhost:8000/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_prediction():
    data = {
        "loanamount": 250000,
        "totaldue": 312500,
        # ... all fields
    }
    response = requests.post("http://localhost:8000/api/v1/predict", json=data)
    assert response.status_code == 200
    assert "default_probability" in response.json()
```

**Run tests:**
```bash
# Start API first
uvicorn src.api.main:app &

# Run tests
python test_api.py
```

---

### Day 15: Documentation

**Add descriptions to endpoints:**

```python
@router.post("/predict",
    response_model=PredictionResponse,
    summary="Predict credit risk",
    description="Returns default probability and risk assessment")
def predict(request: PredictionRequest):
    """
    Predict credit default risk for a loan application.

    **Input:** Customer data and loan details
    **Output:** Risk score, category, and recommendation
    """
    return predictor.predict(request)
```

**Create API README** with:
- Usage examples
- Request/response schemas
- Error codes
- Authentication (if added)

---

## Code Structure After Week 3

```
src/api/
├── __init__.py
├── main.py           # FastAPI app & config
├── routes.py         # Endpoint definitions
├── predictor.py      # ML inference service
└── schemas.py        # Pydantic models

test_api.py           # API tests
```

---

## Testing Your API

### 1. Interactive Docs (Swagger)
Go to: http://localhost:8000/docs

- Try out endpoints directly
- See request/response schemas
- No code needed!

### 2. cURL
```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
-H "Content-Type: application/json" \
-d '{
  "loanamount": 250000,
  "totaldue": 312500,
  "termdays": 180,
  "employment_status_clients": "Permanent",
  "level_of_education_clients": "HND/BSc"
}'
```

### 3. Python
```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/predict",
    json={"loanamount": 250000, ...}
)
print(response.json())
```

---

## Success Criteria

By end of Week 3, you should have:

- [ ] FastAPI application running locally
- [ ] All 5 endpoints working
- [ ] Pydantic validation active
- [ ] Tests passing
- [ ] Swagger docs accessible
- [ ] <100ms prediction latency

---

## Troubleshooting

**"Model file not found"**
- Run Week 2 training first
- Check `models/artifacts/` has model files

**"Validation error"**
- Check request matches schema exactly
- All required fields must be present

**"Import errors"**
- `pip install fastapi uvicorn pydantic`

---

## Next Week Preview

**Week 4: Deployment**
- Dockerize your API
- Deploy to Railway/Render
- Get a live URL
- Show it to employers!

---

**Ready?** Start the API server:
```bash
uvicorn src.api.main:app --reload
```

Then visit: http://localhost:8000/docs 🚀
