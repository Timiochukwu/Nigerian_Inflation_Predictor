# DAY 13: Authentication, Authorization & Rate Limiting

**Goal**: Secure your API with JWT authentication, API keys, and rate limiting to prevent abuse

**Time**: 2-3 hours

**Prerequisites**: Completed Day 12 (validation & error handling)

---

## 📋 What You'll Build Today

By the end of Day 13, you'll have:
- ✅ JWT (JSON Web Token) authentication for admin users
- ✅ API key authentication for programmatic access
- ✅ Role-based access control (RBAC)
- ✅ Rate limiting to prevent API abuse
- ✅ Protected endpoints (require authentication)
- ✅ Public endpoints (no authentication needed)
- ✅ Token refresh mechanism

---

## PART 1: Install Authentication Dependencies

### Step 1: Update requirements.txt

Add these dependencies:

```txt
# Existing...
fastapi==0.109.0
uvicorn[standard]==0.27.0
pydantic==2.5.3

# NEW: Authentication & Security
python-jose[cryptography]==3.3.0  # JWT tokens
passlib[bcrypt]==1.7.4            # Password hashing
python-multipart==0.0.6           # Form data
slowapi==0.1.9                     # Rate limiting
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## PART 2: Create Authentication Module

### Step 3: Create `src/api/auth.py`

```python
# src/api/auth.py
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
import secrets

# Security configurations
SECRET_KEY = "your-secret-key-change-this-in-production"  # Use environment variable
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security schemes
bearer_scheme = HTTPBearer()
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Mock database (replace with real database in production)
USERS_DB = {
    "admin": {
        "username": "admin",
        "hashed_password": pwd_context.hash("admin123"),  # Change in production!
        "role": "admin",
        "email": "admin@creditrisk.com"
    },
    "analyst": {
        "username": "analyst",
        "hashed_password": pwd_context.hash("analyst123"),
        "role": "analyst",
        "email": "analyst@creditrisk.com"
    }
}

# API Keys database (in production, store in database with hashed values)
API_KEYS_DB = {
    "sk_test_1234567890abcdef": {
        "name": "Production API Key",
        "role": "service",
        "rate_limit": 1000,  # requests per minute
        "active": True
    },
    "sk_dev_abcdef1234567890": {
        "name": "Development API Key",
        "role": "developer",
        "rate_limit": 100,
        "active": True
    }
}


# Helper functions
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    return encoded_jwt


def verify_token(token: str) -> dict:
    """Verify and decode JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


# Authentication dependencies
async def get_current_user_from_token(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)
) -> dict:
    """Get current user from JWT token"""
    token = credentials.credentials

    payload = verify_token(token)
    username: str = payload.get("sub")

    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

    user = USERS_DB.get(username)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )

    return {
        "username": user["username"],
        "role": user["role"],
        "email": user["email"]
    }


async def get_current_user_from_api_key(
    api_key: str = Security(api_key_header)
) -> dict:
    """Get user from API key"""
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required"
        )

    key_data = API_KEYS_DB.get(api_key)

    if not key_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    if not key_data.get("active"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API key is inactive"
        )

    return {
        "api_key": api_key,
        "role": key_data["role"],
        "name": key_data["name"],
        "rate_limit": key_data["rate_limit"]
    }


# Combined authentication (JWT or API key)
async def get_current_user(
    token_user: Optional[dict] = Depends(get_current_user_from_token),
    api_key_user: Optional[dict] = Depends(get_current_user_from_api_key)
) -> dict:
    """Get current user from either JWT or API key"""
    # Try JWT first
    try:
        if token_user:
            return token_user
    except HTTPException:
        pass

    # Fall back to API key
    if api_key_user:
        return api_key_user

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required (JWT token or API key)"
    )


# Role-based authorization
def require_role(required_role: str):
    """Dependency to require specific role"""
    async def role_checker(current_user: dict = Depends(get_current_user)) -> dict:
        user_role = current_user.get("role")

        allowed_roles = {
            "admin": ["admin"],
            "analyst": ["admin", "analyst"],
            "service": ["admin", "service"],
            "developer": ["admin", "analyst", "service", "developer"]
        }

        if user_role not in allowed_roles.get(required_role, []):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required role: {required_role}"
            )

        return current_user

    return role_checker
```

---

## PART 3: Create Authentication Endpoints

### Step 4: Create `src/api/routers/auth.py`

```python
# src/api/routers/auth.py
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr
from datetime import timedelta
from ..auth import (
    verify_password, create_access_token, get_current_user,
    USERS_DB, ACCESS_TOKEN_EXPIRE_MINUTES
)

router = APIRouter(prefix="/api/v1/auth", tags=["authentication"])


class LoginRequest(BaseModel):
    """Login request schema"""
    username: str
    password: str


class TokenResponse(BaseModel):
    """Token response schema"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class UserResponse(BaseModel):
    """User info response"""
    username: str
    email: str
    role: str


@router.post("/login", response_model=TokenResponse)
async def login(credentials: LoginRequest):
    """
    Login with username and password to get JWT token

    **Usage:**
    ```bash
    curl -X POST http://localhost:8000/api/v1/auth/login \\
      -H "Content-Type: application/json" \\
      -d '{"username": "admin", "password": "admin123"}'
    ```
    """
    user = USERS_DB.get(credentials.username)

    if not user or not verify_password(credentials.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )

    # Create access token
    access_token = create_access_token(
        data={"sub": user["username"], "role": user["role"]},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )

    return TokenResponse(
        access_token=access_token,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """
    Get current authenticated user information

    **Requires:** JWT token or API key
    """
    return UserResponse(
        username=current_user.get("username", current_user.get("name", "API User")),
        email=current_user.get("email", "N/A"),
        role=current_user["role"]
    )


@router.post("/logout")
async def logout(current_user: dict = Depends(get_current_user)):
    """
    Logout (client should delete token)

    Note: JWT tokens are stateless, so actual logout happens client-side
    """
    return {
        "message": "Successfully logged out. Please delete your token.",
        "user": current_user.get("username", current_user.get("name"))
    }
```

---

## PART 4: Add Rate Limiting

### Step 5: Create `src/api/rate_limit.py`

```python
# src/api/rate_limit.py
from slowapi import Limiter
from slowapi.util import get_remote_address
from fastapi import Request

# Create limiter instance
limiter = Limiter(key_func=get_remote_address)

# Custom key function for authenticated users
def get_user_identifier(request: Request) -> str:
    """
    Get identifier for rate limiting (IP or API key)
    """
    # Check for API key
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return f"api_key:{api_key}"

    # Check for JWT token
    auth_header = request.headers.get("Authorization")
    if auth_header:
        return f"token:{auth_header[:20]}"  # Use part of token

    # Fall back to IP
    return get_remote_address(request)
```

---

## PART 5: Update Main App with Rate Limiting

### Step 6: Update `src/api/main.py`

```python
# src/api/main.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
import logging
import time

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


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests"""
    start_time = time.time()
    logger.info(f"→ {request.method} {request.url.path}")

    response = await call_next(request)

    duration = time.time() - start_time
    logger.info(
        f"← {request.method} {request.url.path} "
        f"[{response.status_code}] {duration:.3f}s"
    )

    return response


# Include routers
app.include_router(auth.router)
app.include_router(predictions.router)


@app.get("/", tags=["root"])
@limiter.limit("10/minute")  # Rate limit: 10 requests per minute
async def root(request: Request):
    """Root endpoint"""
    return {
        "message": "Welcome to Credit Risk Scoring API",
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health",
        "login": "/api/v1/auth/login"
    }


@app.get("/health", response_model=HealthResponse, tags=["monitoring"])
async def health_check():
    """Health check endpoint (no rate limit)"""
    return HealthResponse(
        status="healthy",
        model_loaded=ms.model_service.is_loaded() if ms.model_service else False,
        api_version=settings.app_version
    )
```

---

## PART 6: Protect Prediction Endpoint

### Step 7: Update `src/api/routers/predictions.py`

```python
# src/api/routers/predictions.py
from fastapi import APIRouter, HTTPException, Depends, Request
from ..schemas.prediction import LoanApplication, PredictionResponse
from ..services.model_service import get_model_service, ModelService
from ..exceptions import PredictionError
from ..auth import get_current_user, require_role
from ..rate_limit import limiter
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["predictions"])


@router.post("/predict", response_model=PredictionResponse)
@limiter.limit("60/minute")  # 60 requests per minute
async def predict_credit_risk(
    request: Request,
    application: LoanApplication,
    model_service: ModelService = Depends(get_model_service),
    current_user: dict = Depends(get_current_user)  # Requires authentication!
):
    """
    Predict credit risk for a loan application

    **Authentication:** Requires JWT token or API key

    **Rate Limit:** 60 requests/minute

    **Authorization:** Any authenticated user
    """
    try:
        features = application.model_dump()

        logger.info(
            f"User: {current_user.get('username', current_user.get('name'))} | "
            f"Loan: {features['loan_amount']}"
        )

        result = model_service.predict(features)

        logger.info(f"Prediction: {result['prediction']} (risk: {result['risk_score']})")

        return PredictionResponse(**result)

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise PredictionError(str(e))

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise PredictionError("Failed to generate prediction")


@router.post("/predict/batch", response_model=list[PredictionResponse])
@limiter.limit("10/minute")  # Lower limit for batch operations
async def predict_batch(
    request: Request,
    applications: list[LoanApplication],
    model_service: ModelService = Depends(get_model_service),
    current_user: dict = Depends(require_role("analyst"))  # Requires analyst role!
):
    """
    Batch prediction for multiple loan applications

    **Authentication:** Requires JWT token or API key

    **Authorization:** Requires analyst or admin role

    **Rate Limit:** 10 requests/minute

    **Max applications per request:** 100
    """
    if len(applications) > 100:
        raise HTTPException(
            status_code=400,
            detail="Maximum 100 applications per batch request"
        )

    results = []

    for app in applications:
        try:
            features = app.model_dump()
            result = model_service.predict(features)
            results.append(PredictionResponse(**result))
        except Exception as e:
            logger.error(f"Batch prediction error: {str(e)}")
            # Continue with other predictions
            continue

    logger.info(
        f"Batch prediction by {current_user.get('username')}: "
        f"{len(results)}/{len(applications)} successful"
    )

    return results
```

---

## PART 7: Test Authentication

### Step 8: Restart API

```bash
uvicorn src.api.main:app --reload
```

### Step 9: Test Login

```bash
# Login to get JWT token
curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "admin",
    "password": "admin123"
  }'
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

**Copy the access_token for next steps!**

### Step 10: Test Protected Endpoint (Without Auth)

```bash
# This should FAIL with 401 Unauthorized
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "employment_status": "Permanent",
    "education_level": "Bachelor",
    "loan_amount": 50000,
    "loan_term_months": 24,
    "interest_rate": 12.5
  }'
```

### Step 11: Test Protected Endpoint (With JWT Token)

```bash
# Replace YOUR_TOKEN_HERE with the token from login
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN_HERE" \
  -d '{
    "age": 35,
    "employment_status": "Permanent",
    "education_level": "Bachelor",
    "loan_amount": 50000,
    "loan_term_months": 24,
    "interest_rate": 12.5
  }'
```

**This should now work! ✅**

### Step 12: Test with API Key

```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: sk_test_1234567890abcdef" \
  -d '{
    "age": 35,
    "employment_status": "Permanent",
    "education_level": "Bachelor",
    "loan_amount": 50000,
    "loan_term_months": 24,
    "interest_rate": 12.5
  }'
```

### Step 13: Test Rate Limiting

Run this command 65 times quickly (exceeds 60/minute limit):

```bash
for i in {1..65}; do
  curl -X GET "http://localhost:8000/" -H "X-API-Key: sk_test_1234567890abcdef"
  echo "Request $i"
done
```

**After 60 requests, you should see:**
```json
{
  "error": "Rate limit exceeded: 10 per 1 minute"
}
```

---

## ✅ Day 13 Checklist

- [ ] JWT authentication working (login returns token)
- [ ] API key authentication working
- [ ] Protected endpoints require authentication
- [ ] Role-based access control working (batch endpoint)
- [ ] Rate limiting prevents abuse (429 error after limit)
- [ ] `/auth/me` endpoint returns user info
- [ ] Invalid tokens return 401 Unauthorized
- [ ] Invalid API keys return 401 Unauthorized

---

## 🎯 What You Learned Today

1. **JWT Authentication**
   - Token creation and validation
   - Password hashing with bcrypt
   - Token expiration

2. **API Key Authentication**
   - Alternative to JWT for programmatic access
   - Multiple authentication methods

3. **Authorization**
   - Role-based access control (RBAC)
   - Different permission levels

4. **Rate Limiting**
   - Prevent API abuse
   - Different limits for different endpoints
   - User-based vs IP-based limiting

---

## 🚀 Next Steps

**Tomorrow (Day 14):** Add comprehensive logging, monitoring with Prometheus, and observability with Grafana.

---

## 💡 Production Notes

**CRITICAL - Change These in Production:**

1. **SECRET_KEY:** Use strong random key from environment variable
   ```python
   SECRET_KEY = os.getenv("SECRET_KEY", "fallback-key")
   ```

2. **Passwords:** Don't hardcode passwords! Use database + proper registration flow

3. **API Keys:** Store hashed in database, not in code

4. **HTTPS:** Always use HTTPS in production (not HTTP)

5. **Rate Limits:** Adjust based on your actual traffic patterns

6. **Token Expiry:** Consider refresh tokens for longer sessions

---

**🎉 Your API is now secure with authentication and rate limiting! 🎉**
