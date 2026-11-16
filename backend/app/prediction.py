import time
from typing import Dict, Any, Optional
import numpy as np

from .config import CLASS_NAMES, MEDICAL_DISCLAIMER
from .model_loader import get_model_manager
from .preprocessing import preprocess_image_bytes
from .schemas import PredictionResponse


def run_prediction_pipeline(image_bytes: bytes, filename: str = "scan.jpg") -> PredictionResponse:
    """
    Executes strict lung CT anatomical validation, neural network inference,
    lung cancer classification, and clinical staging protocol assessment.
    """
    start_time = time.perf_counter()

    # Step 1: Preprocess image bytes (strictly enforces lung-only CT scan anatomy)
    tensor_batch, image_metadata = preprocess_image_bytes(image_bytes, filename=filename)

    # Step 2: Run inference via ModelManager (raises 503 if model is not loaded)
    model_mgr = get_model_manager()
    raw_preds = model_mgr.predict(tensor_batch)

    # raw_preds shape is (1, 3) with softmax probabilities for ['Benign', 'Malignant', 'Normal']
    if len(raw_preds.shape) == 2:
        probs = raw_preds[0]
    else:
        probs = raw_preds.flatten()

    inference_ms = (time.perf_counter() - start_time) * 1000.0

    # Map class probabilities
    class_probs: Dict[str, float] = {
        class_name: round(float(probs[idx]), 4)
        for idx, class_name in enumerate(CLASS_NAMES)
    }

    p_benign = float(probs[0])
    p_malignant = float(probs[1])
    p_normal = float(probs[2])

    top_class_idx = int(np.argmax(probs))
    raw_class = CLASS_NAMES[top_class_idx]
    max_prob = float(probs[top_class_idx])

    # Sorted probabilities to check margin / ambiguity
    sorted_probs = np.sort(probs)
    margin = float(sorted_probs[-1] - sorted_probs[-2])

    # Clinical Inconclusive Criteria:
    # 1. Maximum confidence is below 60% threshold, OR
    # 2. Probability difference between top two classes is very small (< 0.15) indicating classifier ambiguity
    is_inconclusive = (max_prob < 0.60) or (margin < 0.15 and max_prob < 0.70)

    if is_inconclusive:
        detection_result = "Inconclusive / Unable to Determine"
        predicted_class = "Inconclusive / Unable to Determine"
        is_cancer_detected = False
        cancer_status = "Indeterminate / Low Diagnostic Confidence"
        predicted_stage = None
        stage_note = None
        abnormality_desc = (
            "Sub-optimal diagnostic clarity or equivocal parenchymal attenuation pattern. "
            "Feature extraction is unable to reliably differentiate between benign inflammatory "
            "changes and early neoplastic changes on this single CT slice."
        )
        scan_summary = (
            f"Algorithmic uncertainty threshold triggered ({round(max_prob * 100, 1)}% maximum probability, "
            f"margin: {round(margin * 100, 1)}%). The CT scan findings cannot be conclusively established."
        )
        next_steps = (
            "Obtain a repeat thin-slice high-resolution chest CT (HRCT) with standard inspiratory breath-hold protocol. "
            "Mandatory formal review by a board-certified thoracic radiologist."
        )

    elif raw_class == "Malignant":
        detection_result = "Lung Cancer Detected"
        predicted_class = "Lung Cancer Detected"
        is_cancer_detected = True
        cancer_status = "Positive for Malignant Lesion"
        
        # Clinical Staging Guideline:
        # A single CT scan image alone cannot reliably establish the complete cancer stage.
        # Stage classification requires TNM protocol (histopathology, PET-CT, distant imaging).
        predicted_stage = "Stage Classification Requires Further Clinical Evaluation"
        stage_note = (
            "Clinical Staging Protocol: A single axial CT slice cannot establish definitive TNM staging (Stage I–IV). "
            "Complete staging requires 18F-FDG PET-CT for metabolic activity, mediastinal nodal sampling (EBUS/mediastinoscopy), "
            "and contrast-enhanced brain/abdominal MRI to evaluate distant metastases."
        )
        abnormality_desc = (
            "Focal pulmonary mass/lesion exhibiting irregular, spiculated margins and soft-tissue architectural distortion "
            "characteristic of a malignant pulmonary neoplasm."
        )
        scan_summary = (
            f"Lung cancer detected with {round(max_prob * 100, 1)}% model confidence. "
            f"The neural network identified strong morphological features of malignancy on the evaluated axial thoracic slice."
        )
        next_steps = (
            "Immediate referral to a thoracic oncology multidisciplinary team. Recommended diagnostic workup: "
            "diagnostic contrast-enhanced chest/upper abdomen CT, 18F-FDG PET-CT, and CT-guided core biopsy or "
            "bronchoscopy for histological subtyping (NSCLC vs SCLC) and molecular biomarker testing (EGFR, ALK, PD-L1)."
        )

    elif raw_class == "Benign":
        detection_result = "No Lung Cancer Detected"
        predicted_class = "No Lung Cancer Detected"
        is_cancer_detected = False
        cancer_status = "Negative for Malignancy (Benign Pulmonary Nodule)"
        predicted_stage = None
        stage_note = None
        abnormality_desc = (
            "Well-circumscribed nodular density displaying smooth, demarcated margins without infiltrative parenchymal invasion, "
            "consistent with a non-malignant benign pulmonary nodule (e.g., granuloma, hamartoma)."
        )
        scan_summary = (
            f"No lung cancer detected. The model identified non-malignant benign nodular features with "
            f"{round(max_prob * 100, 1)}% confidence (benign probability {round(p_benign * 100, 1)}%)."
        )
        next_steps = (
            "Follow-up Low-Dose CT (LDCT) in 3 to 6 months per Fleischner Society pulmonary nodule management guidelines "
            "to document volumetric stability. Consult a pulmonologist if respiratory symptoms persist."
        )

    else:  # Normal
        detection_result = "No Lung Cancer Detected"
        predicted_class = "No Lung Cancer Detected"
        is_cancer_detected = False
        cancer_status = "Negative for Malignancy (Normal Lung Parenchyma)"
        predicted_stage = None
        stage_note = None
        abnormality_desc = (
            "Normal pulmonary parenchymal attenuation with symmetric bronchovascular arborization and clear pleural spaces. "
            "No evidence of solitary pulmonary nodules, masses, or parenchymal consolidation."
        )
        scan_summary = (
            f"No lung cancer detected. The model confirmed clear, healthy lung parenchyma with "
            f"{round(max_prob * 100, 1)}% confidence (Normal probability {round(p_normal * 100, 1)}%)."
        )
        next_steps = (
            "Continue routine preventive pulmonary care and tobacco avoidance. Annual low-dose CT screening is recommended "
            "only for individuals meeting USPSTF high-risk criteria (age 50–80 years with a 20 pack-year smoking history)."
        )

    return PredictionResponse(
        success=True,
        predicted_class=predicted_class,
        detection_result=detection_result,
        cancer_status=cancer_status,
        is_cancer_detected=is_cancer_detected,
        predicted_stage=predicted_stage,
        stage_classification_note=stage_note,
        confidence=round(max_prob, 4),
        raw_class=raw_class,
        class_probabilities=class_probs,
        abnormality_description=abnormality_desc,
        scan_summary=scan_summary,
        recommended_next_steps=next_steps,
        inference_time_ms=round(inference_ms, 2),
        image_meta=image_metadata,
        disclaimer=MEDICAL_DISCLAIMER
    )
