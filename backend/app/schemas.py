from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., json_schema_extra={"example": "healthy"})
    backend_version: str = Field(..., json_schema_extra={"example": "1.0.0"})
    model_loaded: bool = Field(..., json_schema_extra={"example": True})
    model_name: Optional[str] = Field(None, json_schema_extra={"example": "LungGuard_AI_EfficientNetB0"})
    model_path: str = Field(..., json_schema_extra={"example": "models/LungGuard_AI_Final.keras"})
    model_file_exists: bool = Field(..., json_schema_extra={"example": True})
    tensorflow_installed: bool = Field(..., json_schema_extra={"example": True})
    device: str = Field(..., json_schema_extra={"example": "CPU"})
    timestamp: str = Field(..., json_schema_extra={"example": "2026-11-08T00:00:00Z"})


class ModelInfoResponse(BaseModel):
    model_name: str
    source_notebook: str
    dataset: str
    classes: List[str]
    input_shape: List[int]
    architecture: Dict[str, Any]
    training_hyperparameters: Dict[str, Any]
    dataset_split: Dict[str, Any]
    verified_evaluation: Dict[str, Any]
    clinical_and_validation_limitations: List[str]
    disclaimer: str


class PredictionResponse(BaseModel):
    success: bool = Field(True, json_schema_extra={"example": True})
    test_id: Optional[str] = Field(None, json_schema_extra={"example": "LUNG-CT-20261108-A1B2"})
    predicted_class: str = Field(..., json_schema_extra={"example": "Lung Cancer Detected"})
    detection_result: str = Field(..., json_schema_extra={"example": "Lung Cancer Detected"})
    cancer_status: str = Field(..., json_schema_extra={"example": "Positive for Malignant Lesion"})
    is_cancer_detected: bool = Field(..., json_schema_extra={"example": True})
    predicted_stage: Optional[str] = Field(None, json_schema_extra={"example": "Stage Classification Requires Further Clinical Evaluation"})
    stage_classification_note: Optional[str] = None
    confidence: float = Field(..., json_schema_extra={"example": 0.9458})
    raw_class: str = Field(..., json_schema_extra={"example": "Malignant"})
    class_probabilities: Dict[str, float] = Field(
        ...,
        json_schema_extra={"example": {"Benign": 0.021, "Malignant": 0.9458, "Normal": 0.033}}
    )
    abnormality_description: str
    scan_summary: str
    recommended_next_steps: str
    inference_time_ms: float = Field(..., json_schema_extra={"example": 42.5})
    image_meta: Dict[str, Any] = Field(
        default_factory=dict,
        json_schema_extra={"example": {"original_size": [512, 512], "model_input_size": [224, 224, 3], "channels": 3}}
    )
    disclaimer: str = Field(
        ...,
        json_schema_extra={"example": "This is an AI research prototype, not a medical diagnostic device."}
    )


class BreathingAssessmentRequest(BaseModel):
    assessment_type: str = Field(..., description="guided_exercise or spirometry_pft")
    duration_sec: Optional[int] = 0
    completed_cycles: Optional[int] = 0
    breath_hold_sec: Optional[float] = 0.0
    fev1: Optional[float] = None
    fvc: Optional[float] = None
    ratio: Optional[float] = None
    pef: Optional[float] = None
    notes: Optional[str] = ""


class BreathingAssessmentResponse(BaseModel):
    success: bool = True
    test_id: str
    created_at: str
    test_type: str = "Breathing Test"
    assessment_type: str
    result_title: str
    interpretation: str
    clinical_values: Dict[str, Any]
    disclaimer: str


class DashboardStatsResponse(BaseModel):
    total_ct_scans: int
    cancer_detected: int
    no_cancer_detected: int
    inconclusive: int
    total_breathing_assessments: int
    total_tests: int
    daily_activity: List[Dict[str, Any]]


class SampleScan(BaseModel):
    id: str
    filename: str
    class_label: str
    title: str
    description: str
    relative_path: str
    image_url: str


class ErrorResponse(BaseModel):
    success: bool = False
    detail: str
    error_code: str
    hint: Optional[str] = None
