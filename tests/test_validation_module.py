import glob
import io
import os
import pytest
from fastapi import HTTPException
from PIL import Image
import numpy as np

from backend.app.validation import (
    process_and_validate_ct_image,
    validate_and_classify_radiology_image
)


def test_all_authentic_sample_ct_scans_pass():
    """All genuine IQ-OTH-NCCD chest CT scans must pass validation."""
    cts = glob.glob("frontend/assets/sample_*.jpg")
    assert len(cts) >= 10, f"Expected at least 10 sample scans, found {len(cts)}"
    for path in cts:
        with open(path, "rb") as f:
            data = f.read()
        img, meta = process_and_validate_ct_image(data, os.path.basename(path))
        assert meta["status"] == "verified_lung_ct"
        assert img.size[0] >= 128


def test_reject_normal_color_photo():
    """Color photographs and illustrations must be rejected."""
    im = Image.new("RGB", (300, 300), color=(220, 50, 40))
    buf = io.BytesIO()
    im.save(buf, format="JPEG")
    data = buf.getvalue()

    with pytest.raises(HTTPException) as exc:
        process_and_validate_ct_image(data, "photo.jpg")
    assert exc.value.status_code == 400
    assert "Invalid image. Only Lung CT scans are accepted." in exc.value.detail


def test_reject_poor_quality_small_resolution():
    """Low resolution scans (< 128x128) must be rejected."""
    im = Image.new("RGB", (64, 64), color=(50, 50, 50))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    data = buf.getvalue()

    with pytest.raises(HTTPException) as exc:
        process_and_validate_ct_image(data, "tiny.png")
    assert exc.value.status_code == 400
    assert "Unable to analyze this scan due to insufficient image quality." in exc.value.detail


def test_reject_poor_quality_blank_image():
    """Blank or flat scans (std dev < 10) must be rejected."""
    im = Image.new("RGB", (256, 256), color=(40, 40, 40))
    buf = io.BytesIO()
    im.save(buf, format="JPEG")
    data = buf.getvalue()

    with pytest.raises(HTTPException) as exc:
        process_and_validate_ct_image(data, "blank.jpg")
    assert exc.value.status_code == 400
    assert "Unable to analyze this scan due to insufficient image quality." in exc.value.detail


def test_reject_brain_ct_simulation():
    """Brain / Head CT with continuous skull ring and solid parenchyma must be rejected."""
    arr = np.full((256, 256), 35, dtype=np.uint8)  # air background
    yy, xx = np.mgrid[:256, :256]
    dist = np.sqrt((yy - 128)**2 + (xx - 128)**2)
    # skull bone ring
    arr[(dist <= 90) & (dist >= 80)] = 220
    # solid brain parenchyma inside skull (no lung air)
    arr[dist < 80] = 145

    rgb = np.stack([arr, arr, arr], axis=-1)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    data = buf.getvalue()

    with pytest.raises(HTTPException) as exc:
        process_and_validate_ct_image(data, "brain_scan.png")
    assert exc.value.status_code == 400
    assert "Invalid scan. Please upload a Lung CT scan." in exc.value.detail


def test_reject_abdomen_ct_simulation():
    """Abdomen CT with solid viscera and absent lung cavities must be rejected."""
    arr = np.full((256, 256), 35, dtype=np.uint8)  # air background
    yy, xx = np.mgrid[:256, :256]
    dist = np.sqrt((yy - 128)**2 + (xx - 128)**2)
    # abdominal body contour filled with solid viscera (HU +40, intensity 150)
    arr[dist <= 95] = 150
    # spine at posterior
    arr[(dist <= 95) & (yy > 170) & (abs(xx - 128) < 20)] = 230

    rgb = np.stack([arr, arr, arr], axis=-1)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    data = buf.getvalue()

    with pytest.raises(HTTPException) as exc:
        process_and_validate_ct_image(data, "abdomen_ct.png")
    assert exc.value.status_code == 400
    assert "This scan does not contain the required lung anatomy." in exc.value.detail


def test_reject_corrupted_file():
    """Corrupted non-image bytes must be rejected."""
    junk = b"NOT_A_VALID_IMAGE_OR_DICOM_BYTES_1234567890"
    with pytest.raises(HTTPException) as exc:
        process_and_validate_ct_image(junk, "corrupt.jpg")
    assert exc.value.status_code == 400
    assert "Cannot identify image file" in exc.value.detail


def _create_mock_dicom_bytes(modality="CT", body_part="CHEST", pixel_array=None) -> bytes:
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, CTImageStorage

    if pixel_array is None:
        pixel_array = np.full((128, 128), 100, dtype=np.uint16)

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = Dataset()
    ds.file_meta = file_meta
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.Modality = modality
    ds.BodyPartExamined = body_part
    ds.StudyDescription = f"{body_part} SCAN"
    ds.SeriesDescription = f"{body_part} AXIAL"
    ds.PatientName = "TEST^PATIENT"
    ds.Rows, ds.Columns = pixel_array.shape
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelData = pixel_array.tobytes()

    buf = io.BytesIO()
    pydicom.dcmwrite(buf, ds, write_like_original=False)
    return buf.getvalue()


def test_reject_mri_dicom():
    """DICOM with MR modality must be rejected as MRI."""
    dcm_bytes = _create_mock_dicom_bytes(modality="MR", body_part="BRAIN")
    with pytest.raises(HTTPException) as exc:
        process_and_validate_ct_image(dcm_bytes, "scan.dcm")
    assert exc.value.status_code == 400
    assert "MRI scans are not supported." in exc.value.detail


def test_reject_xray_dicom():
    """DICOM with DX/CR/XR modality must be rejected as X-ray."""
    dcm_bytes = _create_mock_dicom_bytes(modality="DX", body_part="CHEST")
    with pytest.raises(HTTPException) as exc:
        process_and_validate_ct_image(dcm_bytes, "chest_xray.dcm")
    assert exc.value.status_code == 400
    assert "X-ray images are not supported. Please upload a Lung CT scan." in exc.value.detail


def test_reject_brain_dicom():
    """DICOM CT with Head/Brain metadata must be rejected."""
    dcm_bytes = _create_mock_dicom_bytes(modality="CT", body_part="BRAIN")
    with pytest.raises(HTTPException) as exc:
        process_and_validate_ct_image(dcm_bytes, "brain.dcm")
    assert exc.value.status_code == 400
    assert "Invalid scan. Please upload a Lung CT scan." in exc.value.detail


def test_reject_abdomen_dicom():
    """DICOM CT with Abdomen metadata must be rejected."""
    dcm_bytes = _create_mock_dicom_bytes(modality="CT", body_part="ABDOMEN")
    with pytest.raises(HTTPException) as exc:
        process_and_validate_ct_image(dcm_bytes, "abdomen.dcm")
    assert exc.value.status_code == 400
    assert "This scan does not contain the required lung anatomy." in exc.value.detail

