"""
Configuration settings for the FastAPI backend
"""

from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Model configuration
MODEL_DIR = BASE_DIR / "model"
MODEL_FILE = "logistic_model.joblib"
MODEL_PATH = MODEL_DIR / MODEL_FILE

# API configuration
API_TITLE = "Titanic Survival Prediction API"
API_DESCRIPTION = "A REST API for predicting Titanic passenger survival using Logistic Regression"
API_VERSION = "1.0.0"

# CORS configuration
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:8080",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8080",
    "*"  # Allow all origins in development
]

# Server configuration
HOST = "0.0.0.0"
PORT = 8000
