"""
LungGuard AI — Strict Lung-Only CT Scan Validation Engine
Implements clinically rigorous validation to ensure ONLY authentic, compatible
chest CT scans containing visible lung anatomy enter the AI cancer detection pipeline.

Rejection Categories & Exact Clinically Approved Messages:
- Brain / Head CT: "Invalid scan. Please upload a Lung CT scan."
- Abdomen / Pelvis CT: "This scan does not contain the required lung anatomy."
- Spine / Bone CT: "This scan does not contain the required lung anatomy."
- Chest X-Ray: "X-ray images are not supported. Please upload a Lung CT scan."
- MRI Scan: "MRI scans are not supported."
- Normal Photo / Animal / Random: "Invalid image. Only Lung CT scans are accepted."
- Poor Quality / Blurry / Corrupted: "Unable to analyze this scan due to insufficient image quality."
- Uncertain Image: "Unable to verify this as a compatible Lung CT scan."
"""

import io
from typing import Tuple, Dict, Any, Optional
import numpy as np
from PIL import Image, UnidentifiedImageError
from fastapi import HTTPException, status

try:
    import pydicom
    from pydicom.errors import InvalidDicomError
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False


# Supported file extensions
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".dcm"}

# Magic byte signatures
JPEG_MAGIC = b"\xFF\xD8\xFF"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
DICOM_PREFIX = b"DICM"


def is_dicom_bytes(contents: bytes) -> bool:
    """
    Checks if raw bytes represent a DICOM file.
    Standard DICOM files have 'DICM' at offset 128.
    Some raw stream files start directly with DICOM group/element tags.
    """
    if len(contents) >= 132 and contents[128:132] == DICOM_PREFIX:
        return True
    if len(contents) >= 4 and contents[:2] in (b"\x02\x00", b"\x08\x00"):
        return True
    return False


def validate_dicom_dataset(ds: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Validates DICOM metadata and extracts calibrated 2D pixel array.
    """
    # 1. Modality Validation
    modality = str(getattr(ds, "Modality", "")).strip().upper()
    
    if modality == "MR":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="MRI scans are not supported."
        )
    if modality in ("CR", "DX", "XR", "RG"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="X-ray images are not supported. Please upload a Lung CT scan."
        )
    if modality == "US":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid scan. Please upload a Lung CT scan."
        )
    if modality and modality != "CT":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid scan. Please upload a Lung CT scan."
        )

    # 2. Body Part / Study Description Metadata Check
    body_part = str(getattr(ds, "BodyPartExamined", "")).upper()
    study_desc = str(getattr(ds, "StudyDescription", "")).upper()
    series_desc = str(getattr(ds, "SeriesDescription", "")).upper()
    protocol = str(getattr(ds, "ProtocolName", "")).upper()
    combined_desc = f"{body_part} {study_desc} {series_desc} {protocol}"

    # Brain / Head rejection
    if any(k in combined_desc for k in ["HEAD", "BRAIN", "SKULL", "CRANIUM", "NEURO"]):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid scan. Please upload a Lung CT scan."
        )

    # Abdomen / Pelvis rejection
    if any(k in combined_desc for k in ["ABDOMEN", "PELVIS", "LIVER", "KIDNEY", "BOWEL", "COLON", "ABDOMINAL"]):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This scan does not contain the required lung anatomy."
        )

    # Spine / Extremity rejection
    if any(k in combined_desc for k in ["SPINE", "LUMBAR", "CERVICAL", "SACRUM"]) and not any(k in combined_desc for k in ["CHEST", "THORAX", "LUNG"]):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This scan does not contain the required lung anatomy."
        )
    if any(k in combined_desc for k in ["EXTREMITY", "LEG", "ARM", "HAND", "FOOT", "KNEE", "ANKLE", "WRIST"]):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid scan. Please upload a Lung CT scan."
        )

    # 3. Pixel Array Extraction & Windowing
    try:
        pixel_array = ds.pixel_array.astype(np.float32)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unable to extract pixel data from DICOM file: {str(exc)}"
        )

    # If multi-frame, take middle slice
    if pixel_array.ndim == 3 and pixel_array.shape[0] > 1:
        pixel_array = pixel_array[pixel_array.shape[0] // 2]
    elif pixel_array.ndim == 3 and pixel_array.shape[-1] in (1, 3):
        if pixel_array.shape[-1] == 3:
            pixel_array = np.mean(pixel_array, axis=-1)
        else:
            pixel_array = pixel_array[:, :, 0]

    # Apply Hounsfield Slope / Intercept
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    hu = pixel_array * slope + intercept

    # Standard Chest CT Lung Window (WW: 1500, WL: -600 -> Range: -1350 to +150 HU)
    window_min = -1350.0
    window_max = 150.0
    clipped = np.clip(hu, window_min, window_max)
    normalized = ((clipped - window_min) / (window_max - window_min) * 255.0).astype(np.uint8)

    # Convert to RGB uint8 array for standard pipeline
    rgb_arr = np.stack([normalized, normalized, normalized], axis=-1)

    meta = {
        "dicom_modality": modality or "CT",
        "dicom_body_part": body_part or "CHEST",
        "hu_min": float(np.min(hu)),
        "hu_max": float(np.max(hu)),
        "hu_mean": float(np.mean(hu))
    }
    return rgb_arr, meta


def validate_and_classify_radiology_image(
    rgb_arr: np.ndarray,
    metadata_hint: Optional[str] = None
) -> Dict[str, Any]:
    """
    Performs comprehensive anatomical and morphological validation on the 2D image.
    Ensures the image is an authentic axial chest CT slice containing bilateral lung fields.
    """
    h, w, c = rgb_arr.shape

    # 1. Quality & Dimension Checks
    if h < 128 or w < 128:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to analyze this scan due to insufficient image quality."
        )

    # 2. Color Saturation & Chromatic Divergence Check
    # Authentic chest CT scans are monochromatic radiological acquisitions.
    r = rgb_arr[:, :, 0].astype(np.float32)
    g = rgb_arr[:, :, 1].astype(np.float32)
    b = rgb_arr[:, :, 2].astype(np.float32)

    channel_diff = float(np.mean(np.abs(r - g) + np.abs(g - b) + np.abs(r - b)) / 3.0)
    max_channel_diff = float(np.max(np.abs(r - g) + np.abs(g - b) + np.abs(r - b)) / 3.0)

    # Non-grayscale images are photographs, web artwork, or non-CT diagrams
    if channel_diff > 8.0 or max_channel_diff > 28.0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image. Only Lung CT scans are accepted."
        )

    # Work with grayscale intensity
    gray = np.mean(rgb_arr, axis=-1).astype(np.float32)

    # 3. Contrast, Standard Deviation & Saturation Checks
    std_dev = float(np.std(gray))
    if std_dev < 10.0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to analyze this scan due to insufficient image quality."
        )

    # Extreme overexposure or underexposure
    overexposed_ratio = float(np.mean(gray > 250.0))
    underexposed_ratio = float(np.mean(gray < 5.0))
    if overexposed_ratio > 0.85 or underexposed_ratio > 0.96:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to analyze this scan due to insufficient image quality."
        )

    # Simple blur check (discrete 2nd derivative approximation)
    laplacian = (
        np.roll(gray, 1, axis=0) + np.roll(gray, -1, axis=0) +
        np.roll(gray, 1, axis=1) + np.roll(gray, -1, axis=1) - 4 * gray
    )
    laplacian_var = float(np.var(laplacian[1:-1, 1:-1]))
    if laplacian_var < 5.0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to analyze this scan due to insufficient image quality."
        )

    # 4. CT Spatial Gantry Background Check
    # In an axial CT scan, the corners represent the air bed / gantry outside the patient.
    corner_h, corner_w = max(4, h // 10), max(4, w // 10)
    tl = np.mean(gray[:corner_h, :corner_w])
    tr = np.mean(gray[:corner_h, -corner_w:])
    bl = np.mean(gray[-corner_h:, :corner_w])
    br = np.mean(gray[-corner_h:, -corner_w:])
    corner_mean = float(np.mean([tl, tr, bl, br]))

    # Grayscale everyday photos / landscapes / documents have bright or non-uniform corners
    if corner_mean > 75.0:
        # Check if projection X-ray or general photo
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image. Only Lung CT scans are accepted."
        )

    # 5. Patient Body Segmentation & Area
    body_thresh = corner_mean + 12.0
    body_mask = gray > body_thresh
    body_ratio = float(np.mean(body_mask))

    # Patient cross-section in an axial scan typically occupies 22% - 85% of the frame
    if body_ratio < 0.15:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to analyze this scan due to insufficient image quality."
        )
    if body_ratio > 0.96 and corner_mean > 50.0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image. Only Lung CT scans are accepted."
        )

    # 6. Anatomical Classification: Brain CT vs Abdomen CT vs Lung CT
    # In a Lung CT, the lung fields are large, dark, bilateral air cavities (HU < -500, pixel intensity < 118).
    # In contrast:
    # - Brain CT: Skull ring enclosing solid brain tissue; dark fraction inside brain is < 6%.
    # - Abdomen CT: Solid parenchymal organs (liver, spleen, kidney) filling the peritoneal cavity; dark air < 10%.
    lung_air_mask = body_mask & (gray < 118.0)
    body_pixels = np.sum(body_mask)
    lung_air_ratio = float(np.sum(lung_air_mask) / (body_pixels + 1e-5))

    # Check for Skull Bone Ring (Brain / Head CT)
    # The human calvarium is a prominent, high-density oval ring (pixel intensity > 205).
    high_bone_mask = body_mask & (gray > 205.0)
    bone_ratio = float(np.sum(high_bone_mask) / (body_pixels + 1e-5))

    # Outer perimeter of body contour for ring check
    if lung_air_ratio < 0.10:
        # Less than 10% dark air inside body -> NOT a lung scan!
        # Differentiate between Brain CT and Abdomen CT:
        if bone_ratio > 0.08:
            # Thick bony calvarium surrounding solid brain tissue
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid scan. Please upload a Lung CT scan."
            )
        else:
            # Solid abdominal/pelvic viscera or non-lung anatomy
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="This scan does not contain the required lung anatomy."
            )

    # 7. Bilateral Lung Cavity Morphology & Symmetry
    # The lungs are paired organs located in the left and right hemithorax.
    mid_x = w // 2
    left_lung_pixels = float(np.sum(lung_air_mask[:, :mid_x]))
    right_lung_pixels = float(np.sum(lung_air_mask[:, mid_x:]))

    if left_lung_pixels < 100 or right_lung_pixels < 100:
        # Isolated single cavity or unilateral artifact
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This scan does not contain the required lung anatomy."
        )

    bilateral_symmetry = min(left_lung_pixels, right_lung_pixels) / (max(left_lung_pixels, right_lung_pixels) + 1e-5)
    if bilateral_symmetry < 0.25:
        # Extreme asymmetry inconsistent with paired thoracic lung fields
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This scan does not contain the required lung anatomy."
        )

    # 8. Mediastinum Verification
    # The space between left and right lungs contains the heart, trachea, and spine (higher density).
    mid_strip_w = max(4, w // 16)
    mid_col = gray[:, mid_x - mid_strip_w : mid_x + mid_strip_w]
    mid_body_mask = body_mask[:, mid_x - mid_strip_w : mid_x + mid_strip_w]
    if np.any(mid_body_mask):
        mid_density = float(np.mean(mid_col[mid_body_mask]))
        if mid_density < 90.0:
            # If the center is hollow/empty air instead of mediastinum, not a thoracic scan
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unable to verify this as a compatible Lung CT scan."
            )

    # 9. Projection X-Ray / Planar Radiography Detection
    # Chest X-rays have continuous horizontal rib projections crossing the lung fields,
    # whereas axial CT has discrete rib dots around the periphery.
    # Also, chest X-rays have a prominent convex diaphragm boundary at bottom and clavicles at top.
    if lung_air_ratio > 0.72:
        # In axial CT, lungs rarely exceed 65% of the body cross-section due to chest wall and mediastinum.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="X-ray images are not supported. Please upload a Lung CT scan."
        )

    return {
        "status": "verified_lung_ct",
        "body_ratio": body_ratio,
        "lung_air_ratio": lung_air_ratio,
        "bilateral_symmetry": bilateral_symmetry,
        "corner_mean": corner_mean
    }


def process_and_validate_ct_image(contents: bytes, filename: str) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Main entry point for strict lung CT validation.
    Accepts raw file bytes and filename. Returns validated PIL Image (RGB) ready for AI preprocessing.
    """
    if not contents or len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is empty (0 bytes)."
        )

    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext and ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file format '{ext}'. Only Lung CT images (.jpg, .jpeg, .png, .dcm) are allowed."
        )

    # 1. Handle DICOM Files
    if ext == ".dcm" or is_dicom_bytes(contents):
        if not PYDICOM_AVAILABLE:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="DICOM processing library is currently unavailable on server."
            )
        try:
            ds = pydicom.dcmread(io.BytesIO(contents), force=True)
            rgb_arr, dicom_meta = validate_dicom_dataset(ds)
            # Run image-level anatomical checks
            anatomy_meta = validate_and_classify_radiology_image(rgb_arr, metadata_hint=filename)
            combined_meta = {**dicom_meta, **anatomy_meta, "original_mode": "DICOM"}
            pil_img = Image.fromarray(rgb_arr)
            return pil_img, combined_meta
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unable to parse DICOM CT scan: {str(exc)}"
            )

    # 2. Handle Standard Image Files (JPEG / PNG)
    # Magic bytes check
    if not (contents.startswith(JPEG_MAGIC) or contents.startswith(PNG_MAGIC)):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot identify image file. The uploaded image file appears corrupted or unreadable."
        )

    try:
        image_stream = io.BytesIO(contents)
        with Image.open(image_stream) as img:
            img.load()
            orig_mode = img.mode
            if orig_mode != "RGB":
                rgb_img = img.convert("RGB")
            else:
                rgb_img = img.copy()

        rgb_arr = np.array(rgb_img)
        meta = validate_and_classify_radiology_image(rgb_arr, metadata_hint=filename)
        meta["original_mode"] = orig_mode
        return rgb_img, meta

    except HTTPException:
        raise
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot identify image file. The uploaded image file appears corrupted or unreadable."
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Image validation failed: {str(exc)}"
        )
