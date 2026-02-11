"""
Application configuration settings.
"""
import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Application
    APP_NAME: str = "Credit Risk Scoring API"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = "Production-grade credit risk prediction system"
    DEBUG: bool = False

    # API
    API_V1_PREFIX: str = "/api/v1"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4

    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    MODELS_DIR: Path = BASE_DIR / "models"
    ARTIFACTS_DIR: Path = MODELS_DIR / "artifacts"
    LOGS_DIR: Path = BASE_DIR / "logs"

    # Model
    MODEL_NAME: str = "credit_risk_xgboost"
    MODEL_VERSION: str = "v1"
    MODEL_PATH: Optional[Path] = None
    SCALER_PATH: Optional[Path] = None

    # ML Parameters
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.2
    CV_FOLDS: int = 5

    # Monitoring
    LOG_LEVEL: str = "INFO"
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090

    # Performance
    MAX_BATCH_SIZE: int = 1000
    PREDICTION_TIMEOUT: int = 30  # seconds

    class Config:
        env_file = ".env"
        case_sensitive = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Set model paths based on model name and version
        if not self.MODEL_PATH:
            self.MODEL_PATH = self.ARTIFACTS_DIR / f"{self.MODEL_NAME}_{self.MODEL_VERSION}.pkl"
        if not self.SCALER_PATH:
            self.SCALER_PATH = self.ARTIFACTS_DIR / f"scaler_{self.MODEL_VERSION}.pkl"

        # Create directories if they don't exist
        for directory in [
            self.DATA_DIR, self.RAW_DATA_DIR, self.PROCESSED_DATA_DIR,
            self.MODELS_DIR, self.ARTIFACTS_DIR, self.LOGS_DIR
        ]:
            directory.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
