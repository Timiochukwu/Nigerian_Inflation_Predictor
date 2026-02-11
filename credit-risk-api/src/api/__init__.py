"""API package for credit risk prediction."""
from .main import app
from .predictor import get_predictor

__all__ = ["app", "get_predictor"]
