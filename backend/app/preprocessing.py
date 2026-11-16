import io
from typing import Tuple, Dict, Any
from PIL import Image, UnidentifiedImageError
import numpy as np
from fastapi import HTTPException, UploadFile, status

from .config import (
    IMAGE_SIZE,
    TARGET_CHANNELS,
    MAX_UPLOAD_SIZE_BYTES,
    ALLOWED_EXTENSIONS
)
from .validation import (
    process_and_validate_ct_image,
    is_dicom_bytes,
    JPEG_MAGIC,
    PNG_MAGIC
)


def validate_file_metadata(file: UploadFile) -> str:
    """
    Validates file extension and basic filename security.
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a valid filename."
        )

    ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file format '{ext}'. Only Lung CT images (.jpg, .jpeg, .png, .dcm) are allowed."
        )

    return ext


def validate_magic_bytes(header: bytes) -> str:
    """
    Validates file header magic bytes to prevent spoofed file extensions.
    """
    if header.startswith(JPEG_MAGIC):
        return "image/jpeg"
    elif header.startswith(PNG_MAGIC):
        return "image/png"
    elif is_dicom_bytes(header) or (len(header) >= 4 and header[:2] in (b"\x02\x00", b"\x08\x00")):
        return "application/dicom"
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot identify image file. The uploaded image file appears corrupted or unreadable."
        )


async def read_and_validate_upload(file: UploadFile) -> Tuple[bytes, str]:
    """
    Reads the file stream with a strict size limit and checks image magic bytes.
    """
    validate_file_metadata(file)

    # Read bytes safely
    contents = await file.read()
    if not contents or len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is empty (0 bytes)."
        )

    if len(contents) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} MB."
        )

    mime = validate_magic_bytes(contents[:256])
    return contents, mime


def preprocess_image_bytes(image_bytes: bytes, filename: str = "scan.jpg") -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Strict Lung CT Validation and Preprocessing:
    1. Validates that image is an authentic, compatible chest CT scan with visible lung anatomy.
    2. Rejects non-lung CTs (Brain, Abdomen, Spine, X-Ray, MRI, Normal Photos, Blurry, Corrupted).
    3. Resizes to (224, 224) and formats as (1, 224, 224, 3) float32 tensor [0.0, 255.0] for EfficientNetB0.
    """
    try:
        # Strictly validate that this is an authentic Lung CT scan
        pil_img, validation_meta = process_and_validate_ct_image(image_bytes, filename)

        orig_width, orig_height = pil_img.size
        orig_mode = validation_meta.get("original_mode", pil_img.mode)

        # Resize to model input shape (224, 224)
        resized_img = pil_img.resize(IMAGE_SIZE, resample=Image.Resampling.LANCZOS)
        img_array = np.asarray(resized_img, dtype=np.float32)

        # Add batch dimension: (1, 224, 224, 3)
        tensor_batch = np.expand_dims(img_array, axis=0)

        metadata = {
            "original_dimensions": [orig_width, orig_height],
            "original_mode": orig_mode,
            "model_input_shape": list(tensor_batch.shape),
            "channels": TARGET_CHANNELS,
            "pixel_range": [float(np.min(tensor_batch)), float(np.max(tensor_batch))],
            **validation_meta
        }

        return tensor_batch, metadata

    except HTTPException:
        raise
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot identify image file. The uploaded image file appears corrupted or unreadable."
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Image processing failed: {str(exc)}"
        )
