"""
LungGuard AI — Model Verification Utility

Run this script to verify that the EfficientNetB0 model file is correctly placed,
loads properly in TensorFlow/Keras, and matches expected input/output shapes.
"""

import sys
import os
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_MODEL_PATH = SCRIPT_DIR / "LungGuard_AI_Final.keras"
METADATA_PATH = SCRIPT_DIR / "model_metadata.json"

MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))


def verify_model():
    print("=" * 65)
    print("      LUNGGUARD AI — MODEL INTEGRATION VERIFICATION")
    print("=" * 65)
    print(f"Target model path: {MODEL_PATH.resolve()}")

    if not MODEL_PATH.exists():
        print("\n[!] MODEL FILE NOT FOUND")
        print("-" * 65)
        print("The trained model 'LungGuard_AI_Final.keras' is not in the models/ folder.")
        print("\nStep-by-step Instructions:")
        print("1. Open your Google Drive where the Colab notebook saved the models:")
        print("   Google Drive Path: /content/drive/MyDrive/LungGuard_AI/Models/")
        print("2. Download 'LungGuard_AI_Final.keras' (or 'lungguard_best.keras').")
        print("   Note: As verified in Cell 44 of the notebook, 'LungGuard_AI_Final.keras'")
        print("   is the 94.58% test-accuracy model copied from 'lungguard_best.keras'.")
        print("3. Place the file at:")
        print(f"   {MODEL_PATH.resolve()}")
        print("-" * 65)
        return False

    print("\n[+] Model file found on disk.")
    file_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"    File size: {file_size_mb:.2f} MB (Expected ~16.3 MB)")

    # Test loading with TensorFlow
    print("\n[*] Checking TensorFlow installation...")
    try:
        import tensorflow as tf
        import numpy as np
        print(f"    TensorFlow Version: {tf.__version__}")
    except ImportError:
        print("\n[!] TensorFlow is not installed in the active Python environment.")
        print("    Install it via: pip install tensorflow>=2.15.0")
        return False

    print("\n[*] Loading Keras model...")
    try:
        model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
        print("    Model successfully loaded into memory!")
    except Exception as exc:
        print(f"\n[!] Failed to load model file: {exc}")
        return False

    # Check input and output shapes
    input_shape = model.input_shape
    output_shape = model.output_shape
    print(f"    Input shape:  {input_shape} (Expected: (None, 224, 224, 3))")
    print(f"    Output shape: {output_shape} (Expected: (None, 3))")

    expected_classes = ["Benign", "Malignant", "Normal"]
    print(f"    Output classes: {expected_classes}")

    if input_shape != (None, 224, 224, 3) or output_shape != (None, 3):
        print("\n[!] Warning: Model shape does not match expected (None, 224, 224, 3) -> (None, 3)")
        return False

    # Perform a test inference with random tensor
    print("\n[*] Testing forward inference pass...")
    dummy_input = np.ones((1, 224, 224, 3), dtype=np.float32) * 128.0
    preds = model.predict(dummy_input, verbose=0)[0]
    print(f"    Test predictions: Benign={preds[0]:.4f}, Malignant={preds[1]:.4f}, Normal={preds[2]:.4f}")
    print(f"    Sum of probabilities: {np.sum(preds):.4f}")

    print("\n" + "=" * 65)
    print(" [+] MODEL VERIFICATION SUCCESSFUL! READY FOR PRODUCTION INFERENCE.")
    print("=" * 65)
    return True


if __name__ == "__main__":
    success = verify_model()
    sys.exit(0 if success else 1)
