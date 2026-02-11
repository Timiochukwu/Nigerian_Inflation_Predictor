"""
API routes for credit risk prediction endpoints.
"""
from fastapi import APIRouter, HTTPException, Depends, status
from typing import List
import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import settings
from src.api.schemas import (
    PredictionRequest, PredictionResponse,
    BatchPredictionRequest, BatchPredictionResponse,
    HealthResponse, ModelInfoResponse
)
from src.api.predictor import get_predictor, CreditRiskPredictor


# Create router
router = APIRouter(prefix=settings.API_V1_PREFIX, tags=["predictions"])


@router.post("/predict",
             response_model=PredictionResponse,
             status_code=status.HTTP_200_OK,
             summary="Predict credit risk for single application",
             description="Returns default probability and risk assessment for a single loan application")
async def predict(
    request: PredictionRequest,
    predictor: CreditRiskPredictor = Depends(get_predictor)
) -> PredictionResponse:
    """
    Make a single credit risk prediction.

    **Input:** Customer demographics, current loan details, and historical loan behavior

    **Output:** Default probability, risk category, and recommended action
    """
    try:
        start_time = time.time()

        # Make prediction
        response = predictor.predict(request)

        # Log prediction time (for monitoring)
        prediction_time = time.time() - start_time
        if prediction_time > 1.0:  # Log if > 1 second
            print(f"⚠️  Slow prediction: {prediction_time:.3f}s")

        return response

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post("/predict/batch",
             response_model=BatchPredictionResponse,
             status_code=status.HTTP_200_OK,
             summary="Predict credit risk for multiple applications",
             description="Batch prediction endpoint for processing multiple loan applications")
async def predict_batch(
    batch_request: BatchPredictionRequest,
    predictor: CreditRiskPredictor = Depends(get_predictor)
) -> BatchPredictionResponse:
    """
    Make batch credit risk predictions.

    **Input:** List of prediction requests (max 1000)

    **Output:** List of predictions with metadata
    """
    try:
        start_time = time.time()

        # Validate batch size
        if len(batch_request.requests) > settings.MAX_BATCH_SIZE:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Batch size exceeds maximum allowed ({settings.MAX_BATCH_SIZE})"
            )

        # Make predictions
        predictions = predictor.predict_batch(batch_request.requests)

        # Create response
        response = BatchPredictionResponse(
            predictions=predictions,
            total_processed=len(predictions)
        )

        # Log batch processing time
        processing_time = time.time() - start_time
        print(f"✅ Batch prediction: {len(predictions)} requests in {processing_time:.3f}s")

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@router.get("/health",
            response_model=HealthResponse,
            status_code=status.HTTP_200_OK,
            summary="Health check endpoint",
            description="Check if API and model are ready")
async def health_check(
    predictor: CreditRiskPredictor = Depends(get_predictor)
) -> HealthResponse:
    """
    Health check endpoint for monitoring and readiness probes.

    Returns API status and model availability.
    """
    model_loaded = predictor.model is not None and predictor.scaler is not None

    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        model_version=settings.MODEL_VERSION
    )


@router.get("/model/info",
            response_model=ModelInfoResponse,
            status_code=status.HTTP_200_OK,
            summary="Get model information",
            description="Returns model metadata, performance metrics, and feature importance")
async def model_info(
    predictor: CreditRiskPredictor = Depends(get_predictor)
) -> ModelInfoResponse:
    """
    Get detailed model information.

    Includes performance metrics, feature importance, and model metadata.
    """
    try:
        info = predictor.get_model_info()

        return ModelInfoResponse(
            model_name=info['model_name'],
            model_version=info['model_version'],
            n_features=info['n_features'],
            performance_metrics=info['performance_metrics'],
            top_features=info['top_features'],
            last_trained="2024-07-15"  # In production, load from metadata
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve model info: {str(e)}"
        )
