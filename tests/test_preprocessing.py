import io
import pytest
import numpy as np
from PIL import Image
from fastapi import HTTPException

from backend.app.preprocessing import (
    preprocess_image_bytes,
    validate_magic_bytes
)


def create_test_image(size=(300, 300), fmt="JPEG"):
    sample_path = "frontend/assets/sample_benign_bengin_case_(1).jpg"
    with open(sample_path, "rb") as f:
        img = Image.open(f)
        if size:
            img = img.resize(size)
        buf = io.BytesIO()
        img.save(buf, format=fmt)
        return buf.getvalue()


def test_magic_bytes_detection():
    jpeg_bytes = create_test_image(fmt="JPEG")
    png_bytes = create_test_image(fmt="PNG")

    assert validate_magic_bytes(jpeg_bytes[:16]) == "image/jpeg"
    assert validate_magic_bytes(png_bytes[:16]) == "image/png"

    with pytest.raises(HTTPException) as exc:
        validate_magic_bytes(b"INVALID_HEADER_DATA")
    assert exc.value.status_code == 400


def test_preprocessing_dimensions_and_dtype():
    raw_bytes = create_test_image(size=(450, 600), fmt="JPEG")
    tensor, meta = preprocess_image_bytes(raw_bytes)

    # Must be batch of 1 with shape (1, 224, 224, 3)
    assert tensor.shape == (1, 224, 224, 3)
    assert tensor.dtype == np.float32

    # Verification of pixel values: raw [0, 255] range for EfficientNetB0
    assert meta["model_input_shape"] == [1, 224, 224, 3]
    assert meta["channels"] == 3
    assert meta["original_dimensions"] == [450, 600]
    assert np.max(tensor) <= 255.0
    assert np.min(tensor) >= 0.0


def test_grayscale_image_rgb_conversion():
    # Grayscale image (Mode 'L') from authentic sample CT
    sample_path = "frontend/assets/sample_benign_bengin_case_(1).jpg"
    with open(sample_path, "rb") as f:
        img = Image.open(f).convert("L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    raw_bytes = buf.getvalue()

    tensor, meta = preprocess_image_bytes(raw_bytes)
    assert tensor.shape == (1, 224, 224, 3)
    assert meta["original_mode"] == "L"


def test_corrupt_image_rejection():
    # Corrupt data with correct header
    fake_corrupt = b"\xFF\xD8\xFF" + b"CorruptGarbageDataNotAnImage" * 10
    with pytest.raises(HTTPException) as exc:
        preprocess_image_bytes(fake_corrupt)
    assert exc.value.status_code in [400, 422]
