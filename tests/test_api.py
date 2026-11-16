import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.main import app
from backend.app.model_loader import get_model_manager

client = TestClient(app)


def create_sample_jpeg():
    # Use authentic IQ-OTH-NCCD lung CT sample scan
    sample_path = "frontend/assets/sample_benign_bengin_case_(1).jpg"
    with open(sample_path, "rb") as f:
        return f.read()


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "backend_version" in data
    assert data["backend_version"] == "1.0.0"
    assert "model_loaded" in data
    assert "device" in data


def test_model_info_endpoint():
    response = client.get("/api/v1/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["classes"] == ["Benign", "Malignant", "Normal"]
    assert data["input_shape"] == [224, 224, 3]
    assert "verified_evaluation" in data
    assert data["verified_evaluation"]["original_best_model"]["test_accuracy"] == 0.9458


def test_samples_endpoint():
    response = client.get("/api/v1/samples")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    first = data[0]
    assert "id" in first
    assert "class_label" in first
    assert "image_url" in first


def test_predict_invalid_extension():
    response = client.post(
        "/predict",
        files={"file": ("test.txt", b"plain text data", "text/plain")}
    )
    assert response.status_code == 415


def test_predict_empty_file():
    response = client.post(
        "/predict",
        files={"file": ("empty.jpg", b"", "image/jpeg")}
    )
    assert response.status_code == 400


def test_predict_when_model_file_absent(monkeypatch):
    # Ensure model is marked as not loaded and file does not exist
    from pathlib import Path
    mgr = get_model_manager()
    monkeypatch.setattr(mgr, "is_loaded", False)
    monkeypatch.setattr(mgr, "model_path", Path("non_existent_model_file.keras"))
    monkeypatch.setattr(
        mgr,
        "load_error",
        "Model file not found at 'non_existent_model_file.keras'."
    )

    sample_bytes = create_sample_jpeg()
    response = client.post(
        "/predict",
        files={"file": ("ct_scan.jpg", sample_bytes, "image/jpeg")}
    )
    # Must fail cleanly with 503 Service Unavailable, NEVER return fake prediction
    assert response.status_code == 503
    assert "Model is currently unavailable" in response.json()["detail"]


def test_predict_with_mocked_model_inference(monkeypatch):
    mgr = get_model_manager()
    monkeypatch.setattr(mgr, "is_loaded", True)
    # Mock predict output for Normal class: [0.01, 0.04, 0.95]
    import numpy as np
    monkeypatch.setattr(mgr, "predict", lambda x: np.array([[0.01, 0.04, 0.95]], dtype=np.float32))

    sample_bytes = create_sample_jpeg()
    response = client.post(
        "/predict",
        files={"file": ("ct_scan.jpg", sample_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["detection_result"] == "No Lung Cancer Detected"
    assert data["is_cancer_detected"] is False
    assert data["predicted_stage"] is None
    assert data["raw_class"] == "Normal"
    assert data["confidence"] == 0.95
    assert data["class_probabilities"]["Normal"] == 0.95
    assert "abnormality_description" in data
    assert "scan_summary" in data
    assert "recommended_next_steps" in data
    assert "disclaimer" in data


def test_predict_lung_cancer_detected(monkeypatch):
    mgr = get_model_manager()
    monkeypatch.setattr(mgr, "is_loaded", True)
    # Mock predict output for Malignant (Lung Cancer): [0.05, 0.92, 0.03]
    import numpy as np
    monkeypatch.setattr(mgr, "predict", lambda x: np.array([[0.05, 0.92, 0.03]], dtype=np.float32))

    sample_bytes = create_sample_jpeg()
    response = client.post(
        "/predict",
        files={"file": ("ct_scan.jpg", sample_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["detection_result"] == "Lung Cancer Detected"
    assert data["is_cancer_detected"] is True
    assert data["cancer_status"] == "Positive for Malignant Lesion"
    assert data["predicted_stage"] == "Stage Classification Requires Further Clinical Evaluation"
    assert "TNM" in data["stage_classification_note"]
    assert data["confidence"] == 0.92


def test_predict_inconclusive_case(monkeypatch):
    mgr = get_model_manager()
    monkeypatch.setattr(mgr, "is_loaded", True)
    # Mock predict output with low confidence/ambiguity: [0.36, 0.34, 0.30]
    import numpy as np
    monkeypatch.setattr(mgr, "predict", lambda x: np.array([[0.36, 0.34, 0.30]], dtype=np.float32))

    sample_bytes = create_sample_jpeg()
    response = client.post(
        "/predict",
        files={"file": ("ct_scan.jpg", sample_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["detection_result"] == "Inconclusive / Unable to Determine"
    assert data["is_cancer_detected"] is False
    assert data["predicted_stage"] is None
    assert "uncertainty" in data["scan_summary"].lower()


def test_reject_vivid_color_non_ct_image():
    # Create an RGB image with high color saturation (e.g. bright red photo)
    buf = io.BytesIO()
    img = Image.new("RGB", (256, 256), color=(255, 10, 10))
    img.save(buf, format="JPEG")
    color_bytes = buf.getvalue()

    response = client.post(
        "/predict",
        files={"file": ("photo.jpg", color_bytes, "image/jpeg")}
    )
    assert response.status_code == 400
    assert "Invalid image. Only Lung CT scans are accepted." in response.json()["detail"]
