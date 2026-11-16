import os
from pathlib import Path
from typing import List

# Base workspace paths
APP_DIR = Path(__file__).resolve().parent
BACKEND_DIR = APP_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent

# Model configuration
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "LungGuard_AI_Final.keras"
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
MODEL_METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
SAMPLES_METADATA_PATH = PROJECT_ROOT / "assets" / "sample_scans" / "metadata.json"
ASSETS_DIR = PROJECT_ROOT / "assets"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

# Image and inference settings
IMAGE_SIZE = (224, 224)
TARGET_CHANNELS = 3
CLASS_NAMES = ["Benign", "Malignant", "Normal"]  # Alphabetical index 0, 1, 2 matching notebook
MAX_UPLOAD_SIZE_BYTES = int(os.getenv("MAX_UPLOAD_SIZE_BYTES", 10 * 1024 * 1024))  # 10 MB limit
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".dcm"}
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/pjpeg", "image/x-png", "application/dicom", "application/octet-stream"}

# Server and CORS settings
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
DEBUG = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")

_cors_env = os.getenv("CORS_ORIGINS", "*")
CORS_ORIGINS: List[str] = [origin.strip() for origin in _cors_env.split(",") if origin.strip()]

# Medical disclaimer constant
MEDICAL_DISCLAIMER = (
    "This is an AI research prototype, not a medical diagnostic device. "
    "Its predictions are not a medical diagnosis. "
    "Please consult a qualified radiologist or healthcare professional for medical interpretation."
)
