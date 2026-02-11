# DAY 14: Logging, Monitoring & Observability

**Goal**: Add production-grade logging, metrics collection with Prometheus, and monitoring dashboards

**Time**: 2-3 hours

**Prerequisites**: Completed Day 13 (authentication & rate limiting)

---

## 📋 What You'll Build Today

By the end of Day 14, you'll have:
- ✅ Structured logging with proper log levels (INFO, WARNING, ERROR)
- ✅ Request/response logging with correlation IDs
- ✅ Prometheus metrics endpoint (`/metrics`)
- ✅ Custom business metrics (predictions, errors, latency)
- ✅ Monitoring dashboard setup (Prometheus + Grafana)
- ✅ Alerting rules for critical issues
- ✅ Log aggregation and analysis

---

## PART 1: Advanced Structured Logging

### Step 1: Update requirements.txt

```txt
# Existing...
fastapi==0.109.0
uvicorn[standard]==0.27.0

# NEW: Logging & Monitoring
python-json-logger==2.0.7     # Structured JSON logging
prometheus-client==0.19.0      # Prometheus metrics
prometheus-fastapi-instrumentator==6.1.0  # FastAPI metrics
```

Install:
```bash
pip install -r requirements.txt
```

---

## PART 2: Set Up Structured Logging

### Step 2: Create `src/api/logging_config.py`

```python
# src/api/logging_config.py
import logging
import sys
from pythonjsonlogger import jsonlogger
from datetime import datetime
import uuid
from contextvars import ContextVar

# Context variable for request ID (persists across async calls)
request_id_var: ContextVar[str] = ContextVar('request_id', default='')


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter that adds request_id and timestamp"""

    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)

        # Add timestamp
        log_record['timestamp'] = datetime.utcnow().isoformat() + 'Z'

        # Add request ID if available
        request_id = request_id_var.get()
        if request_id:
            log_record['request_id'] = request_id

        # Add log level
        log_record['level'] = record.levelname

        # Add logger name
        log_record['logger'] = record.name


def setup_logging(log_level: str = "INFO"):
    """
    Configure structured JSON logging

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Create handler
    handler = logging.StreamHandler(sys.stdout)

    # Set JSON formatter
    formatter = CustomJsonFormatter(
        '%(timestamp)s %(level)s %(name)s %(message)s'
    )
    handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    root_logger.handlers = []  # Remove existing handlers
    root_logger.addHandler(handler)

    # Set level for specific loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)


def generate_request_id() -> str:
    """Generate unique request ID"""
    return str(uuid.uuid4())
```

---

## PART 3: Add Request Tracking Middleware

### Step 3: Create `src/api/middleware.py`

```python
# src/api/middleware.py
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
import time
import logging
from .logging_config import request_id_var, generate_request_id

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to log all requests with correlation IDs
    """

    async def dispatch(self, request: Request, call_next):
        # Generate request ID
        request_id = request.headers.get("X-Request-ID", generate_request_id())
        request_id_var.set(request_id)

        # Start timer
        start_time = time.time()

        # Log incoming request
        logger.info(
            "Incoming request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "client_ip": request.client.host,
                "user_agent": request.headers.get("user-agent", "unknown")
            }
        )

        # Process request
        try:
            response = await call_next(request)

            # Calculate duration
            duration = time.time() - start_time

            # Log response
            logger.info(
                "Request completed",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": round(duration * 1000, 2)
                }
            )

            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id

            return response

        except Exception as e:
            duration = time.time() - start_time

            # Log error
            logger.error(
                "Request failed",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "error": str(e),
                    "duration_ms": round(duration * 1000, 2)
                },
                exc_info=True
            )

            raise
```

---

## PART 4: Add Prometheus Metrics

### Step 4: Create `src/api/metrics.py`

```python
# src/api/metrics.py
from prometheus_client import Counter, Histogram, Gauge, Info
from functools import wraps
import time
import logging

logger = logging.getLogger(__name__)

# Define custom metrics

# Counter: Total predictions made
predictions_total = Counter(
    'credit_risk_predictions_total',
    'Total number of credit risk predictions',
    ['prediction', 'user_role']  # Labels for filtering
)

# Counter: Prediction errors
prediction_errors_total = Counter(
    'credit_risk_prediction_errors_total',
    'Total number of prediction errors',
    ['error_type']
)

# Histogram: Prediction latency
prediction_duration_seconds = Histogram(
    'credit_risk_prediction_duration_seconds',
    'Time spent processing prediction',
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

# Gauge: Model load status
model_loaded = Gauge(
    'credit_risk_model_loaded',
    'Whether the ML model is loaded (1=loaded, 0=not loaded)'
)

# Counter: Authentication attempts
auth_attempts_total = Counter(
    'credit_risk_auth_attempts_total',
    'Total authentication attempts',
    ['method', 'status']  # method: jwt/api_key, status: success/failure
)

# Histogram: Request duration
request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint', 'status_code']
)

# Counter: High-risk predictions
high_risk_predictions = Counter(
    'credit_risk_high_risk_predictions_total',
    'Number of high-risk predictions (risk_score >= 60)',
    ['user_role']
)

# Info: Model metadata
model_info = Info(
    'credit_risk_model',
    'Information about the loaded model'
)


def track_prediction_metrics(func):
    """
    Decorator to track prediction metrics
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()

        try:
            # Execute prediction
            result = await func(*args, **kwargs)

            # Track successful prediction
            duration = time.time() - start_time
            prediction_duration_seconds.observe(duration)

            # Get user role from current_user
            current_user = kwargs.get('current_user', {})
            user_role = current_user.get('role', 'unknown')

            # Track prediction type
            predictions_total.labels(
                prediction=result.prediction,
                user_role=user_role
            ).inc()

            # Track high-risk predictions
            if result.risk_score >= 60:
                high_risk_predictions.labels(user_role=user_role).inc()

            logger.info(
                f"Prediction metric recorded: {result.prediction} "
                f"(risk: {result.risk_score}, duration: {duration:.3f}s)"
            )

            return result

        except Exception as e:
            # Track error
            error_type = type(e).__name__
            prediction_errors_total.labels(error_type=error_type).inc()

            logger.error(f"Prediction error tracked: {error_type}")

            raise

    return wrapper
```

---

## PART 5: Update Main App with Monitoring

### Step 5: Update `src/api/main.py`

```python
# src/api/main.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from prometheus_fastapi_instrumentator import Instrumentator
import logging

from .config import settings
from .routers import predictions, auth
from .schemas.prediction import HealthResponse
from .services import model_service as ms
from .exceptions import (
    validation_exception_handler,
    general_exception_handler,
    prediction_exception_handler,
    PredictionError
)
from .rate_limit import limiter
from .logging_config import setup_logging
from .middleware import RequestLoggingMiddleware
from .metrics import model_loaded, model_info

# Setup structured logging
setup_logging(log_level="INFO")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    logger.info("🚀 Starting Credit Risk API...")

    # Load model
    ms.model_service = ms.ModelService(settings.model_path)

    # Update model metrics
    model_loaded.set(1)
    model_info.info({
        'model_type': 'XGBoost',  # Update based on your model
        'version': '1.0',
        'path': str(settings.model_path)
    })

    logger.info("✅ Model loaded successfully")

    yield

    # Shutdown
    model_loaded.set(0)
    logger.info("👋 Shutting down API...")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=settings.app_description,
    lifespan=lifespan
)

# Add Prometheus instrumentation
instrumentator = Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True,
    should_respect_env_var=True,
    should_instrument_requests_inprogress=True,
    excluded_handlers=["/metrics"],
    env_var_name="ENABLE_METRICS",
    inprogress_name="http_requests_inprogress",
    inprogress_labels=True
)

instrumentator.instrument(app).expose(app, include_in_schema=True, tags=["monitoring"])

# Add custom middleware
app.add_middleware(RequestLoggingMiddleware)

# Add rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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

# Include routers
app.include_router(auth.router)
app.include_router(predictions.router)


@app.get("/", tags=["root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to Credit Risk Scoring API",
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics"
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

## PART 6: Update Prediction Router with Metrics

### Step 6: Update `src/api/routers/predictions.py`

```python
# src/api/routers/predictions.py
from fastapi import APIRouter, HTTPException, Depends, Request
from ..schemas.prediction import LoanApplication, PredictionResponse
from ..services.model_service import get_model_service, ModelService
from ..exceptions import PredictionError
from ..auth import get_current_user
from ..rate_limit import limiter
from ..metrics import track_prediction_metrics
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["predictions"])


@router.post("/predict", response_model=PredictionResponse)
@limiter.limit("60/minute")
@track_prediction_metrics  # Add metrics tracking!
async def predict_credit_risk(
    request: Request,
    application: LoanApplication,
    model_service: ModelService = Depends(get_model_service),
    current_user: dict = Depends(get_current_user)
):
    """
    Predict credit risk for a loan application
    """
    try:
        features = application.model_dump()

        logger.info(
            "Processing prediction",
            extra={
                "user": current_user.get('username', current_user.get('name')),
                "loan_amount": features['loan_amount']
            }
        )

        result = model_service.predict(features)

        logger.info(
            "Prediction successful",
            extra={
                "prediction": result['prediction'],
                "risk_score": result['risk_score'],
                "confidence": result.get('confidence_level')
            }
        )

        return PredictionResponse(**result)

    except ValueError as e:
        logger.error("Validation error", extra={"error": str(e)})
        raise PredictionError(str(e))

    except Exception as e:
        logger.error("Prediction failed", extra={"error": str(e)}, exc_info=True)
        raise PredictionError("Failed to generate prediction")
```

---

## PART 7: Test Metrics Endpoint

### Step 7: Restart API

```bash
uvicorn src.api.main:app --reload
```

### Step 8: Access Metrics Endpoint

Open browser or use curl:

```bash
curl http://localhost:8000/metrics
```

**Expected output:**
```
# HELP credit_risk_predictions_total Total number of credit risk predictions
# TYPE credit_risk_predictions_total counter
credit_risk_predictions_total{prediction="Good",user_role="admin"} 5.0
credit_risk_predictions_total{prediction="Bad",user_role="admin"} 2.0

# HELP credit_risk_prediction_duration_seconds Time spent processing prediction
# TYPE credit_risk_prediction_duration_seconds histogram
credit_risk_prediction_duration_seconds_bucket{le="0.01"} 3.0
credit_risk_prediction_duration_seconds_bucket{le="0.05"} 7.0
credit_risk_prediction_duration_seconds_bucket{le="0.1"} 7.0
credit_risk_prediction_duration_seconds_sum 0.234
credit_risk_prediction_duration_seconds_count 7.0

# HELP credit_risk_model_loaded Whether the ML model is loaded
# TYPE credit_risk_model_loaded gauge
credit_risk_model_loaded 1.0

# HELP http_requests_total Total HTTP requests
# TYPE http_requests_total counter
http_requests_total{method="POST",status="2xx"} 15.0
```

---

## PART 8: Set Up Prometheus (Local)

### Step 9: Create `prometheus.yml`

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'credit-risk-api'
    static_configs:
      - targets: ['host.docker.internal:8000']  # For Docker Desktop
        # Or use 'localhost:8000' if running Prometheus locally
    metrics_path: '/metrics'
```

### Step 10: Run Prometheus with Docker

```bash
docker run -d \
  --name prometheus \
  -p 9090:9090 \
  -v $(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus
```

### Step 11: Access Prometheus UI

Open browser:
```
http://localhost:9090
```

Try these queries:
- `credit_risk_predictions_total` - Total predictions
- `rate(credit_risk_predictions_total[1m])` - Predictions per second
- `credit_risk_prediction_duration_seconds_bucket` - Latency distribution
- `credit_risk_high_risk_predictions_total` - High-risk count

---

## PART 9: Set Up Grafana Dashboard (Optional)

### Step 12: Run Grafana with Docker

```bash
docker run -d \
  --name grafana \
  -p 3000:3000 \
  grafana/grafana
```

### Step 13: Configure Grafana

1. Open browser: `http://localhost:3000`
2. Login: `admin` / `admin` (change password when prompted)
3. Add Data Source:
   - Click "Configuration" → "Data Sources"
   - Click "Add data source"
   - Select "Prometheus"
   - URL: `http://host.docker.internal:9090` (Docker Desktop) or `http://localhost:9090`
   - Click "Save & Test"

### Step 14: Create Dashboard

Create a new dashboard with these panels:

**Panel 1: Total Predictions**
```promql
sum(credit_risk_predictions_total)
```

**Panel 2: Prediction Rate (per minute)**
```promql
rate(credit_risk_predictions_total[1m]) * 60
```

**Panel 3: Prediction Latency (p95)**
```promql
histogram_quantile(0.95,
  rate(credit_risk_prediction_duration_seconds_bucket[5m])
)
```

**Panel 4: Good vs Bad Predictions**
```promql
sum by (prediction) (credit_risk_predictions_total)
```

**Panel 5: Error Rate**
```promql
rate(credit_risk_prediction_errors_total[5m])
```

---

## PART 10: View Structured Logs

### Step 15: Generate Some Traffic

```bash
# Login
TOKEN=$(curl -s -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}' | \
  jq -r '.access_token')

# Make predictions
for i in {1..10}; do
  curl -s -X POST "http://localhost:8000/api/v1/predict" \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d '{
      "age": 35,
      "employment_status": "Permanent",
      "education_level": "Bachelor",
      "loan_amount": 50000,
      "loan_term_months": 24,
      "interest_rate": 12.5
    }' > /dev/null
  echo "Request $i sent"
done
```

### Step 16: View JSON Logs

Check your terminal where the API is running. You should see structured JSON logs:

```json
{
  "timestamp": "2026-02-11T10:30:45.123456Z",
  "level": "INFO",
  "logger": "src.api.routers.predictions",
  "message": "Processing prediction",
  "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "user": "admin",
  "loan_amount": 50000.0
}
```

---

## ✅ Day 14 Checklist

- [ ] Structured JSON logging configured
- [ ] Request IDs generated for all requests
- [ ] Prometheus metrics endpoint working (`/metrics`)
- [ ] Custom business metrics tracking (predictions, errors, latency)
- [ ] Prometheus scraping API successfully
- [ ] Grafana dashboard created (optional)
- [ ] Logs include request_id, timestamp, level, message
- [ ] Metrics show prediction counts, duration, errors

---

## 🎯 What You Learned Today

1. **Structured Logging**
   - JSON log format for machine parsing
   - Request correlation with IDs
   - Contextual logging with extra fields

2. **Prometheus Metrics**
   - Counters, Histograms, Gauges
   - Custom business metrics
   - Metrics endpoint exposure

3. **Observability**
   - Request tracking across logs and metrics
   - Performance monitoring
   - Error tracking

4. **Monitoring Setup**
   - Prometheus configuration
   - Grafana dashboards
   - Alerting (basic)

---

## 🚀 Next Steps

**Tomorrow (Day 15):** Write comprehensive tests with pytest (unit tests, integration tests, load tests).

---

## 📚 Additional Resources

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Tutorials](https://grafana.com/tutorials/)
- [FastAPI Observability](https://fastapi.tiangolo.com/advanced/observability/)

---

## 💡 Production Tips

1. **Log Storage:** Use ELK stack (Elasticsearch, Logstash, Kibana) or CloudWatch
2. **Metrics Storage:** Prometheus with long-term storage (Thanos, Cortex)
3. **Alerting:** Set up alerts for:
   - High error rate (> 1%)
   - High latency (p95 > 500ms)
   - Model not loaded
   - High memory usage
4. **Dashboards:** Create role-specific dashboards (ops, business, data science)

---

**🎉 Your API is now observable with metrics and structured logging! 🎉**
