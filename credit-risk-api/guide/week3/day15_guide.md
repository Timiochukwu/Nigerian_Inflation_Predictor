# DAY 15: Comprehensive Testing with Pytest

**Goal**: Write unit tests, integration tests, and load tests to ensure your API is production-ready

**Time**: 2-3 hours

**Prerequisites**: Completed Day 14 (logging & monitoring)

---

## 📋 What You'll Build Today

By the end of Day 15, you'll have:
- ✅ Unit tests for individual functions (validators, model service)
- ✅ Integration tests for API endpoints
- ✅ Authentication/authorization tests
- ✅ Error handling tests
- ✅ Load tests to verify performance
- ✅ Test fixtures and mocks
- ✅ Code coverage reporting
- ✅ CI-ready test suite

---

## PART 1: Install Testing Dependencies

### Step 1: Update requirements.txt

```txt
# Existing...
fastapi==0.109.0
pytest==7.4.4
httpx==0.26.0  # For TestClient

# NEW: Testing Dependencies
pytest-asyncio==0.23.3        # Async test support
pytest-cov==4.1.0             # Code coverage
pytest-mock==3.12.0           # Mocking
faker==22.0.0                 # Generate fake test data
locust==2.20.0                # Load testing
```

Install:
```bash
pip install -r requirements.txt
```

---

## PART 2: Set Up Test Structure

### Step 2: Create Test Directory Structure

```bash
mkdir -p tests
mkdir -p tests/unit
mkdir -p tests/integration
mkdir -p tests/load
```

Your structure:
```
credit-risk-api/
├── tests/
│   ├── __init__.py
│   ├── conftest.py          ← Pytest fixtures
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_validators.py
│   │   ├── test_model_service.py
│   │   └── test_auth.py
│   ├── integration/
│   │   ├── __init__.py
│   │   ├── test_api_endpoints.py
│   │   ├── test_authentication.py
│   │   └── test_prediction_flow.py
│   └── load/
│       └── locustfile.py
└── pytest.ini
```

---

## PART 3: Configure Pytest

### Step 3: Create `pytest.ini`

```ini
# pytest.ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts =
    -v
    --strict-markers
    --tb=short
    --cov=src
    --cov-report=term-missing
    --cov-report=html
markers =
    unit: Unit tests
    integration: Integration tests
    slow: Slow running tests
    auth: Authentication tests
```

---

## PART 4: Create Test Fixtures

### Step 4: Create `tests/conftest.py`

```python
# tests/conftest.py
import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.api.auth import create_access_token
from datetime import timedelta
import joblib
from pathlib import Path
from unittest.mock import MagicMock

@pytest.fixture(scope="session")
def test_client():
    """Create test client for the FastAPI app"""
    return TestClient(app)


@pytest.fixture(scope="session")
def admin_token():
    """Generate JWT token for admin user"""
    token = create_access_token(
        data={"sub": "admin", "role": "admin"},
        expires_delta=timedelta(hours=1)
    )
    return token


@pytest.fixture(scope="session")
def analyst_token():
    """Generate JWT token for analyst user"""
    token = create_access_token(
        data={"sub": "analyst", "role": "analyst"},
        expires_delta=timedelta(hours=1)
    )
    return token


@pytest.fixture(scope="session")
def valid_api_key():
    """Return valid API key for testing"""
    return "sk_test_1234567890abcdef"


@pytest.fixture(scope="session")
def invalid_api_key():
    """Return invalid API key for testing"""
    return "sk_invalid_key"


@pytest.fixture
def sample_loan_application():
    """Sample loan application data"""
    return {
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


@pytest.fixture
def invalid_loan_application():
    """Invalid loan application (negative amount)"""
    return {
        "age": 35,
        "employment_status": "Permanent",
        "education_level": "Bachelor",
        "loan_amount": -5000.0,  # Invalid: negative
        "loan_term_months": 24,
        "interest_rate": 12.5
    }


@pytest.fixture
def mock_model_service(monkeypatch):
    """Mock model service for testing without actual model"""
    mock_service = MagicMock()

    # Mock predict method
    mock_service.predict.return_value = {
        "prediction": "Good",
        "probability_good": 0.85,
        "probability_bad": 0.15,
        "risk_score": 15,
        "recommendation": "APPROVE - Low risk applicant",
        "confidence_level": "High"
    }

    mock_service.is_loaded.return_value = True

    # Patch the model service
    from src.api.services import model_service
    monkeypatch.setattr(model_service, "model_service", mock_service)

    return mock_service
```

---

## PART 5: Unit Tests

### Step 5: Create `tests/unit/test_validators.py`

```python
# tests/unit/test_validators.py
import pytest
from pydantic import ValidationError
from src.api.schemas.prediction import LoanApplication


@pytest.mark.unit
class TestLoanApplicationValidation:
    """Test Pydantic validators for LoanApplication"""

    def test_valid_loan_application(self, sample_loan_application):
        """Test valid loan application passes validation"""
        app = LoanApplication(**sample_loan_application)

        assert app.age == 35
        assert app.loan_amount == 50000.0
        assert app.employment_status == "Permanent"

    def test_invalid_age_too_young(self):
        """Test age validation: too young"""
        with pytest.raises(ValidationError) as exc_info:
            LoanApplication(
                age=17,  # Invalid: < 18
                employment_status="Student",
                education_level="High School",
                loan_amount=10000,
                loan_term_months=12,
                interest_rate=10.0
            )

        errors = exc_info.value.errors()
        assert any("age" in str(err) for err in errors)

    def test_invalid_age_too_old(self):
        """Test age validation: too old"""
        with pytest.raises(ValidationError) as exc_info:
            LoanApplication(
                age=150,  # Invalid: > 100
                employment_status="Retired",
                education_level="Bachelor",
                loan_amount=10000,
                loan_term_months=12,
                interest_rate=10.0
            )

        errors = exc_info.value.errors()
        assert any("age" in str(err) for err in errors)

    def test_invalid_loan_amount_negative(self):
        """Test loan amount validation: negative value"""
        with pytest.raises(ValidationError):
            LoanApplication(
                age=30,
                employment_status="Permanent",
                education_level="Bachelor",
                loan_amount=-5000,  # Invalid
                loan_term_months=24,
                interest_rate=12.5
            )

    def test_invalid_loan_amount_too_small(self):
        """Test loan amount validation: below minimum"""
        with pytest.raises(ValidationError) as exc_info:
            LoanApplication(
                age=30,
                employment_status="Permanent",
                education_level="Bachelor",
                loan_amount=500,  # Invalid: < 1000
                loan_term_months=24,
                interest_rate=12.5
            )

        errors = exc_info.value.errors()
        assert "Loan amount must be at least 1000" in str(errors)

    def test_invalid_employment_status(self):
        """Test employment status validation"""
        with pytest.raises(ValidationError) as exc_info:
            LoanApplication(
                age=30,
                employment_status="InvalidStatus",  # Invalid
                education_level="Bachelor",
                loan_amount=50000,
                loan_term_months=24,
                interest_rate=12.5
            )

        errors = exc_info.value.errors()
        assert "Invalid employment_status" in str(errors)

    def test_high_debt_to_income_ratio(self):
        """Test debt-to-income ratio validation"""
        with pytest.raises(ValidationError) as exc_info:
            LoanApplication(
                age=30,
                employment_status="Permanent",
                education_level="Bachelor",
                loan_amount=50000,
                loan_term_months=24,
                interest_rate=12.5,
                monthly_income=1000  # Too low, causes high DTI
            )

        errors = exc_info.value.errors()
        assert "Debt-to-income ratio too high" in str(errors)

    def test_employment_status_normalization(self):
        """Test that employment status is normalized (title case)"""
        app = LoanApplication(
            age=30,
            employment_status="permanent",  # lowercase
            education_level="bachelor",     # lowercase
            loan_amount=50000,
            loan_term_months=24,
            interest_rate=12.5
        )

        assert app.employment_status == "Permanent"
        assert app.education_level == "Bachelor"
```

### Step 6: Create `tests/unit/test_auth.py`

```python
# tests/unit/test_auth.py
import pytest
from src.api.auth import (
    verify_password, get_password_hash,
    create_access_token, verify_token
)
from datetime import timedelta
from jose import JWTError


@pytest.mark.unit
class TestAuthFunctions:
    """Test authentication helper functions"""

    def test_password_hashing(self):
        """Test password hashing and verification"""
        password = "test_password_123"

        # Hash password
        hashed = get_password_hash(password)

        # Should not be the same as original
        assert hashed != password

        # Should verify correctly
        assert verify_password(password, hashed) is True

        # Wrong password should fail
        assert verify_password("wrong_password", hashed) is False

    def test_create_and_verify_token(self):
        """Test JWT token creation and verification"""
        data = {"sub": "testuser", "role": "admin"}

        # Create token
        token = create_access_token(data, expires_delta=timedelta(minutes=30))

        # Verify token
        payload = verify_token(token)

        assert payload["sub"] == "testuser"
        assert payload["role"] == "admin"
        assert "exp" in payload

    def test_verify_invalid_token(self):
        """Test that invalid tokens raise exception"""
        with pytest.raises(Exception):  # Should raise HTTPException
            verify_token("invalid.token.here")

    def test_token_expiration(self):
        """Test that expired tokens are rejected"""
        data = {"sub": "testuser"}

        # Create token that expires immediately
        token = create_access_token(data, expires_delta=timedelta(seconds=-1))

        # Should fail verification
        with pytest.raises(Exception):
            verify_token(token)
```

---

## PART 6: Integration Tests

### Step 7: Create `tests/integration/test_api_endpoints.py`

```python
# tests/integration/test_api_endpoints.py
import pytest


@pytest.mark.integration
class TestHealthEndpoint:
    """Test health check endpoint"""

    def test_health_check_success(self, test_client):
        """Test health endpoint returns 200"""
        response = test_client.get("/health")

        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data
        assert "api_version" in data

    def test_root_endpoint(self, test_client):
        """Test root endpoint"""
        response = test_client.get("/")

        assert response.status_code == 200

        data = response.json()
        assert "message" in data
        assert "version" in data


@pytest.mark.integration
class TestMetricsEndpoint:
    """Test Prometheus metrics endpoint"""

    def test_metrics_endpoint_exists(self, test_client):
        """Test metrics endpoint returns 200"""
        response = test_client.get("/metrics")

        assert response.status_code == 200
        assert "credit_risk" in response.text or "http_requests" in response.text
```

### Step 8: Create `tests/integration/test_authentication.py`

```python
# tests/integration/test_authentication.py
import pytest


@pytest.mark.integration
@pytest.mark.auth
class TestAuthentication:
    """Test authentication endpoints and flows"""

    def test_login_success(self, test_client):
        """Test successful login"""
        response = test_client.post(
            "/api/v1/auth/login",
            json={"username": "admin", "password": "admin123"}
        )

        assert response.status_code == 200

        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert "expires_in" in data

    def test_login_invalid_credentials(self, test_client):
        """Test login with wrong password"""
        response = test_client.post(
            "/api/v1/auth/login",
            json={"username": "admin", "password": "wrongpassword"}
        )

        assert response.status_code == 401

    def test_login_nonexistent_user(self, test_client):
        """Test login with non-existent user"""
        response = test_client.post(
            "/api/v1/auth/login",
            json={"username": "nonexistent", "password": "password"}
        )

        assert response.status_code == 401

    def test_get_current_user_with_token(self, test_client, admin_token):
        """Test /auth/me endpoint with valid token"""
        response = test_client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 200

        data = response.json()
        assert data["username"] == "admin"
        assert data["role"] == "admin"

    def test_get_current_user_with_api_key(self, test_client, valid_api_key):
        """Test /auth/me endpoint with valid API key"""
        response = test_client.get(
            "/api/v1/auth/me",
            headers={"X-API-Key": valid_api_key}
        )

        assert response.status_code == 200

        data = response.json()
        assert data["role"] == "service"

    def test_get_current_user_no_auth(self, test_client):
        """Test /auth/me endpoint without authentication"""
        response = test_client.get("/api/v1/auth/me")

        assert response.status_code == 401
```

### Step 9: Create `tests/integration/test_prediction_flow.py`

```python
# tests/integration/test_prediction_flow.py
import pytest


@pytest.mark.integration
class TestPredictionEndpoint:
    """Test prediction endpoint"""

    def test_predict_without_auth(self, test_client, sample_loan_application):
        """Test prediction without authentication fails"""
        response = test_client.post(
            "/api/v1/predict",
            json=sample_loan_application
        )

        assert response.status_code == 401

    def test_predict_with_jwt_token(self, test_client, admin_token, sample_loan_application):
        """Test prediction with valid JWT token"""
        response = test_client.post(
            "/api/v1/predict",
            json=sample_loan_application,
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 200

        data = response.json()
        assert "prediction" in data
        assert data["prediction"] in ["Good", "Bad"]
        assert "probability_good" in data
        assert "probability_bad" in data
        assert "risk_score" in data
        assert "recommendation" in data
        assert "confidence_level" in data

    def test_predict_with_api_key(self, test_client, valid_api_key, sample_loan_application):
        """Test prediction with valid API key"""
        response = test_client.post(
            "/api/v1/predict",
            json=sample_loan_application,
            headers={"X-API-Key": valid_api_key}
        )

        assert response.status_code == 200

    def test_predict_invalid_data(self, test_client, admin_token):
        """Test prediction with invalid data returns 422"""
        invalid_data = {
            "age": 17,  # Too young
            "employment_status": "Permanent",
            "education_level": "Bachelor",
            "loan_amount": -5000,  # Negative
            "loan_term_months": 24,
            "interest_rate": 12.5
        }

        response = test_client.post(
            "/api/v1/predict",
            json=invalid_data,
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 422

        data = response.json()
        assert "error" in data
        assert "errors" in data

    def test_predict_missing_required_fields(self, test_client, admin_token):
        """Test prediction with missing fields"""
        incomplete_data = {
            "age": 35,
            "loan_amount": 50000
            # Missing required fields
        }

        response = test_client.post(
            "/api/v1/predict",
            json=incomplete_data,
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == 422


@pytest.mark.integration
class TestBatchPrediction:
    """Test batch prediction endpoint"""

    def test_batch_predict_requires_analyst_role(self, test_client, analyst_token):
        """Test batch prediction works for analyst role"""
        applications = [
            {
                "age": 30,
                "employment_status": "Permanent",
                "education_level": "Bachelor",
                "loan_amount": 30000,
                "loan_term_months": 24,
                "interest_rate": 10.0
            },
            {
                "age": 40,
                "employment_status": "Self-Employed",
                "education_level": "Master",
                "loan_amount": 60000,
                "loan_term_months": 36,
                "interest_rate": 15.0
            }
        ]

        response = test_client.post(
            "/api/v1/predict/batch",
            json=applications,
            headers={"Authorization": f"Bearer {analyst_token}"}
        )

        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, list)
        assert len(data) <= len(applications)

    def test_batch_predict_too_many_applications(self, test_client, analyst_token):
        """Test batch prediction rejects > 100 applications"""
        # Create 101 applications
        applications = [
            {
                "age": 30,
                "employment_status": "Permanent",
                "education_level": "Bachelor",
                "loan_amount": 30000,
                "loan_term_months": 24,
                "interest_rate": 10.0
            }
        ] * 101

        response = test_client.post(
            "/api/v1/predict/batch",
            json=applications,
            headers={"Authorization": f"Bearer {analyst_token}"}
        )

        assert response.status_code == 400
```

---

## PART 7: Run Tests

### Step 10: Run All Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run specific test file
pytest tests/unit/test_validators.py

# Run with verbose output
pytest -v

# Run and stop on first failure
pytest -x
```

**Expected output:**
```
============================= test session starts ==============================
collected 25 items

tests/unit/test_validators.py ........                                   [ 32%]
tests/unit/test_auth.py ....                                             [ 48%]
tests/integration/test_api_endpoints.py ..                               [ 56%]
tests/integration/test_authentication.py ......                          [ 80%]
tests/integration/test_prediction_flow.py .....                          [100%]

========================== 25 passed in 5.23s ==================================
```

---

## PART 8: Load Testing with Locust

### Step 11: Create `tests/load/locustfile.py`

```python
# tests/load/locustfile.py
from locust import HttpUser, task, between
import json

class CreditRiskAPIUser(HttpUser):
    """Simulate API user behavior"""

    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    token = None

    def on_start(self):
        """Login once when user starts"""
        response = self.client.post(
            "/api/v1/auth/login",
            json={"username": "admin", "password": "admin123"}
        )

        if response.status_code == 200:
            self.token = response.json()["access_token"]

    @task(3)  # Weight: 3x more likely than other tasks
    def predict_loan(self):
        """Make a prediction"""
        if not self.token:
            return

        loan_data = {
            "age": 35,
            "employment_status": "Permanent",
            "education_level": "Bachelor",
            "loan_amount": 50000.0,
            "loan_term_months": 24,
            "interest_rate": 12.5,
            "total_previous_loans": 2,
            "avg_repayment_ratio": 1.1,
            "default_history_count": 0
        }

        self.client.post(
            "/api/v1/predict",
            json=loan_data,
            headers={"Authorization": f"Bearer {self.token}"}
        )

    @task(1)
    def health_check(self):
        """Check API health"""
        self.client.get("/health")

    @task(1)
    def get_user_info(self):
        """Get current user info"""
        if not self.token:
            return

        self.client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {self.token}"}
        )
```

### Step 12: Run Load Test

```bash
# Start your API first
uvicorn src.api.main:app --reload

# In another terminal, run Locust
locust -f tests/load/locustfile.py --host=http://localhost:8000
```

**Open browser:** `http://localhost:8089`

- **Number of users:** 100
- **Spawn rate:** 10 users/second
- **Host:** http://localhost:8000

Click "Start swarming" and watch your API performance!

**Expected metrics:**
- Requests/second: > 100
- Average response time: < 100ms
- Failures: < 1%

---

## PART 9: Coverage Report

### Step 13: View Coverage Report

```bash
# Generate coverage report
pytest --cov=src --cov-report=html

# Open report in browser
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

**Target coverage:** > 80%

---

## ✅ Day 15 Checklist

- [ ] Pytest installed and configured
- [ ] Test fixtures created (conftest.py)
- [ ] Unit tests pass (validators, auth)
- [ ] Integration tests pass (endpoints, authentication)
- [ ] Load tests run successfully
- [ ] Code coverage > 80%
- [ ] All tests documented with docstrings
- [ ] Tests can run in CI/CD pipeline

---

## 🎯 What You Learned Today

1. **Unit Testing**
   - Testing individual functions
   - Pydantic validation testing
   - Mocking dependencies

2. **Integration Testing**
   - Testing full API flows
   - Authentication testing
   - Error handling verification

3. **Load Testing**
   - Simulating user traffic
   - Performance benchmarking
   - Identifying bottlenecks

4. **Test Organization**
   - Fixtures for reusable test data
   - Markers for test categorization
   - Coverage reporting

---

## 🚀 What's Next?

**🎉 Congratulations! You've completed Week 3! 🎉**

You now have a **production-ready Credit Risk Scoring API** with:
- ✅ FastAPI application with prediction endpoints
- ✅ Request validation and error handling
- ✅ JWT and API key authentication
- ✅ Rate limiting
- ✅ Structured logging and Prometheus metrics
- ✅ Comprehensive test suite

**Optional Next Steps:**
1. **Dockerize your API** (create Dockerfile, docker-compose.yml)
2. **Deploy to cloud** (AWS Lambda, Google Cloud Run, Azure)
3. **Set up CI/CD** (GitHub Actions, GitLab CI)
4. **Add more features** (model versioning, A/B testing, feature flags)

---

## 📚 Additional Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [FastAPI Testing Guide](https://fastapi.tiangolo.com/tutorial/testing/)
- [Locust Documentation](https://docs.locust.io/)

---

## 💡 CI/CD Integration

### GitHub Actions Example

Create `.github/workflows/test.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt

      - name: Run tests
        run: |
          pytest --cov=src --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

---

**🎉 Your Credit Risk API is now fully tested and production-ready! 🎉**

**Total Time:** 15 days (3 weeks)
- Week 1: Data preparation
- Week 2: ML model training
- Week 3: API development

**You've built something amazing! 🚀**
