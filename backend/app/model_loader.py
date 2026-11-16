import os
import logging
from pathlib import Path
from typing import Optional, Tuple, Any
import numpy as np
from fastapi import HTTPException, status

from .config import MODEL_PATH, CLASS_NAMES

logger = logging.getLogger("lungguard.model_loader")


class ModelManager:
    """
    Singleton Manager for the LungGuard AI EfficientNetB0 Keras model.
    Loads the model once during application startup lifespan.
    """
    _instance: Optional["ModelManager"] = None

    def __init__(self):
        self.model: Any = None
        self.is_loaded: bool = False
        self.model_path: Path = MODEL_PATH
        self.model_name: str = "LungGuard_AI_EfficientNetB0"
        self.tensorflow_available: bool = False
        self.load_error: Optional[str] = None
        self.device: str = "CPU"
        self._check_tf_availability()

    @classmethod
    def get_instance(cls) -> "ModelManager":
        if cls._instance is None:
            cls._instance = ModelManager()
        return cls._instance

    def _check_tf_availability(self):
        try:
            import tensorflow as tf
            self.tensorflow_available = True
            gpus = tf.config.list_physical_devices("GPU")
            self.device = "GPU" if gpus else "CPU"
        except ImportError:
            self.tensorflow_available = False
            self.device = "Unavailable"

    def load_model(self) -> bool:
        """
        Attempts to load the verified saved .keras model.
        """
        logger.info(f"Target model path: {self.model_path.resolve()}")

        if not self.model_path.exists():
            self.is_loaded = False
            self.load_error = (
                f"Model file not found at '{self.model_path.resolve()}'. "
                "Please download 'LungGuard_AI_Final.keras' from Google Colab / Drive "
                "(/content/drive/MyDrive/LungGuard_AI/Models/LungGuard_AI_Final.keras) "
                f"and place it in '{self.model_path.parent.resolve()}'."
            )
            logger.warning(self.load_error)
            return False

        if not self.tensorflow_available:
            self.is_loaded = False
            self.load_error = (
                "TensorFlow is not installed in the active Python environment. "
                "Run 'pip install tensorflow>=2.15.0' or 'pip install tensorflow-cpu>=2.15.0' to enable inference."
            )
            logger.error(self.load_error)
            return False

        try:
            import tensorflow as tf
            logger.info("Loading Keras model from disk...")
            # Load with compile=False for faster inference-only loading
            self.model = tf.keras.models.load_model(str(self.model_path), compile=False)
            self.is_loaded = True
            self.load_error = None

            # Verify model shape
            input_shape = getattr(self.model, "input_shape", None)
            output_shape = getattr(self.model, "output_shape", None)
            logger.info(f"Model successfully loaded! Input shape: {input_shape}, Output shape: {output_shape}")
            return True

        except Exception as exc:
            self.is_loaded = False
            self.load_error = f"Failed to load model file: {str(exc)}"
            logger.error(self.load_error, exc_info=True)
            return False

    def predict(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Runs model inference on preprocessed image batch tensor.
        Raises 503 if the model is not loaded (never generates fake predictions).
        """
        # If not loaded yet, check if file was recently placed and re-attempt loading
        if not self.is_loaded or self.model is None:
            self._check_tf_availability()
            if self.model_path.exists() and self.tensorflow_available:
                self.load_model()

        if not self.is_loaded or self.model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    f"Model is currently unavailable. {self.load_error or 'Model not loaded.'} "
                    f"Ensure 'LungGuard_AI_Final.keras' is placed at '{self.model_path}'."
                )
            )

        try:
            # Predict batch
            predictions = self.model.predict(input_tensor, verbose=0)
            return predictions
        except Exception as exc:
            logger.error(f"Prediction execution failed: {str(exc)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Inference computation error: {str(exc)}"
            )


# Global helper accessor
def get_model_manager() -> ModelManager:
    return ModelManager.get_instance()
