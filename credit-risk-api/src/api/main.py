"""
FastAPI main application for Credit Risk Scoring API.
Production-ready with error handling, logging, and monitoring.
"""
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from contextlib import asynccontextmanager
import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import settings
from src.api.routes import router as api_router
from src.api.predictor import get_predictor


# Lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for application startup and shutdown.

    Startup: Load model and initialize resources
    Shutdown: Cleanup resources
    """
    # Startup
    print("="*60)
    print(f"🚀 Starting {settings.APP_NAME}")
    print("="*60)

    try:
        # Initialize predictor (loads model)
        predictor = get_predictor()
        print(f"✅ Model loaded successfully")
        print(f"   Model: {settings.MODEL_NAME}")
        print(f"   Version: {settings.MODEL_VERSION}")
        print(f"   Features: {len(predictor.feature_names)}")

        print(f"\n✅ {settings.APP_NAME} is ready!")
        print(f"   API: http://{settings.HOST}:{settings.PORT}")
        print(f"   Docs: http://{settings.HOST}:{settings.PORT}/docs")
        print("="*60)

    except Exception as e:
        print(f"❌ Failed to start application: {e}")
        raise

    yield  # Application runs here

    # Shutdown
    print(f"\n🛑 Shutting down {settings.APP_NAME}")
    print("="*60)


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url=f"{settings.API_V1_PREFIX}/openapi.json"
)


# CORS middleware (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.4f}"
    return response


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle uncaught exceptions."""
    print(f"❌ Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error",
            "type": type(exc).__name__
        }
    )


# Include API routes
app.include_router(api_router)


# Root endpoint
@app.get("/",
         tags=["root"],
         summary="Root endpoint",
         description="Welcome message and API information")
async def root():
    """Root endpoint with API information."""
    return {
        "message": f"Welcome to {settings.APP_NAME}",
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": f"{settings.API_V1_PREFIX}/health",
        "endpoints": {
            "predict": f"{settings.API_V1_PREFIX}/predict",
            "batch_predict": f"{settings.API_V1_PREFIX}/predict/batch",
            "model_info": f"{settings.API_V1_PREFIX}/model/info"
        }
    }


# Run with uvicorn if executed directly
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=settings.WORKERS if not settings.DEBUG else 1
    )
