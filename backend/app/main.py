import os
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from .config import (
    CORS_ORIGINS,
    MODEL_PATH,
    MODEL_METADATA_PATH,
    SAMPLES_METADATA_PATH,
    ASSETS_DIR,
    FRONTEND_DIR,
    PROJECT_ROOT,
    MEDICAL_DISCLAIMER
)
from .model_loader import get_model_manager
from .preprocessing import read_and_validate_upload
from .prediction import run_prediction_pipeline
import uuid
from .db import (
    init_db,
    save_test_record,
    get_dashboard_stats,
    list_test_records,
    get_test_record_by_id,
    delete_test_record
)
from .schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    SampleScan,
    ErrorResponse,
    BreathingAssessmentRequest,
    BreathingAssessmentResponse,
    DashboardStatsResponse
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("lungguard.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load the model once and initialize database
    logger.info("Initializing LungGuard AI Backend & Database...")
    init_db()
    model_mgr = get_model_manager()
    model_mgr.load_model()
    if model_mgr.is_loaded:
        logger.info(f"Model successfully initialized from: {model_mgr.model_path}")
    else:
        logger.warning(f"Backend started without active model: {model_mgr.load_error}")
    yield
    # Shutdown logic if any
    logger.info("Shutting down LungGuard AI Backend.")


app = FastAPI(
    title="LungGuard AI API",
    description=(
        "Production-ready backend for Lung CT Image Classification using EfficientNetB0. "
        "Educational and research prototype for Benign, Malignant, and Normal pulmonary CT analysis."
    ),
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS if CORS_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
async def get_health():
    """
    Health check endpoint returning system status and model loading state.
    """
    model_mgr = get_model_manager()
    if not model_mgr.is_loaded:
        model_mgr._check_tf_availability()
        if model_mgr.model_path.exists() and model_mgr.tensorflow_available:
            model_mgr.load_model()

    status_str = "healthy" if model_mgr.is_loaded else "degraded"

    return HealthResponse(
        status=status_str,
        backend_version="1.0.0",
        model_loaded=model_mgr.is_loaded,
        model_name=model_mgr.model_name if model_mgr.is_loaded else None,
        model_path=str(model_mgr.model_path.relative_to(PROJECT_ROOT) if model_mgr.model_path.is_relative_to(PROJECT_ROOT) else model_mgr.model_path),
        model_file_exists=model_mgr.model_path.exists(),
        tensorflow_installed=model_mgr.tensorflow_available,
        device=model_mgr.device,
        timestamp=datetime.now(timezone.utc).isoformat()
    )


@app.get("/api/v1/model-info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    """
    Returns full verified model architecture, training hyperparameters, and evaluation metrics.
    """
    if not MODEL_METADATA_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model metadata specification file is missing."
        )

    try:
        with open(MODEL_METADATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return ModelInfoResponse(**data)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load model metadata: {str(exc)}"
        )


@app.get("/api/v1/samples", response_model=list[SampleScan], tags=["Dataset"])
async def list_sample_scans():
    """
    Returns verified authentic IQ-OTH-NCCD benchmark CT scans available for quick evaluation.
    """
    if not SAMPLES_METADATA_PATH.exists():
        return []

    try:
        with open(SAMPLES_METADATA_PATH, "r", encoding="utf-8") as f:
            samples_data = json.load(f)

        results = []
        for s in samples_data:
            # URL accessible via static route
            image_url = f"/assets/{s['relative_path'].replace('assets/', '')}"
            results.append(SampleScan(
                id=s["id"],
                filename=s["filename"],
                class_label=s["class_label"],
                title=s["title"],
                description=s["description"],
                relative_path=s["relative_path"],
                image_url=image_url
            ))
        return results
    except Exception as exc:
        logger.error(f"Failed to list samples: {str(exc)}")
        return []


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
@app.post("/api/v1/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_ct_image(file: UploadFile = File(...)):
    """
    Upload a lung CT image (JPEG or PNG) to receive model classification and class-wise probabilities.
    Saves record to persistent testing history.
    """
    image_bytes, mime = await read_and_validate_upload(file)
    prediction_result = run_prediction_pipeline(image_bytes, filename=file.filename or "scan.jpg")

    # Save uploaded image file for historical report inspection
    ext = ".png" if "png" in mime else ".jpg"
    upload_filename = f"scan_{uuid.uuid4().hex[:8]}{ext}"
    uploads_dir = ASSETS_DIR / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    saved_path = uploads_dir / upload_filename
    saved_path.write_bytes(image_bytes)
    image_url = f"assets/uploads/{upload_filename}"

    # Also save to frontend/assets/uploads if separate
    fe_uploads = FRONTEND_DIR / "assets" / "uploads"
    fe_uploads.mkdir(parents=True, exist_ok=True)
    (fe_uploads / upload_filename).write_bytes(image_bytes)

    # Record to persistent SQLite database
    test_id = save_test_record(
        test_type="CT Scan",
        detection_result=prediction_result.detection_result,
        cancer_status=prediction_result.cancer_status,
        cancer_stage=prediction_result.predicted_stage or "N/A",
        confidence=prediction_result.confidence,
        image_url=image_url,
        abnormality_description=prediction_result.abnormality_description,
        scan_summary=prediction_result.scan_summary,
        recommended_next_steps=prediction_result.recommended_next_steps,
        raw_metadata={
            "filename": file.filename,
            "class_probabilities": prediction_result.class_probabilities,
            "inference_time_ms": prediction_result.inference_time_ms,
            "image_meta": prediction_result.image_meta,
            "stage_classification_note": prediction_result.stage_classification_note
        }
    )
    prediction_result.test_id = test_id
    return prediction_result


@app.post("/api/v1/predict-sample/{sample_id}", response_model=PredictionResponse, tags=["Inference"])
async def predict_benchmark_sample(sample_id: str):
    """
    Runs model inference on one of the verified benchmark CT scans included in the project.
    Saves record to persistent testing history.
    """
    if not SAMPLES_METADATA_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Benchmark samples catalog not found."
        )

    with open(SAMPLES_METADATA_PATH, "r", encoding="utf-8") as f:
        samples_data = json.load(f)

    target_sample = next((s for s in samples_data if s["id"] == sample_id), None)
    if not target_sample:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sample scan with ID '{sample_id}' not found."
        )

    sample_file_path = PROJECT_ROOT / target_sample["relative_path"]
    if not sample_file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sample scan file does not exist at '{sample_file_path}'."
        )

    image_bytes = sample_file_path.read_bytes()
    prediction_result = run_prediction_pipeline(image_bytes)

    # Record to persistent SQLite database
    test_id = save_test_record(
        test_type="CT Scan",
        detection_result=prediction_result.detection_result,
        cancer_status=prediction_result.cancer_status,
        cancer_stage=prediction_result.predicted_stage or "N/A",
        confidence=prediction_result.confidence,
        image_url=target_sample.get("image_url", target_sample["relative_path"]),
        abnormality_description=prediction_result.abnormality_description,
        scan_summary=prediction_result.scan_summary,
        recommended_next_steps=prediction_result.recommended_next_steps,
        raw_metadata={
            "sample_id": sample_id,
            "title": target_sample.get("title", ""),
            "class_probabilities": prediction_result.class_probabilities,
            "inference_time_ms": prediction_result.inference_time_ms,
            "stage_classification_note": prediction_result.stage_classification_note
        }
    )
    prediction_result.test_id = test_id
    return prediction_result


# ==========================================
# Breathing Test API
# ==========================================
@app.post("/api/v1/breathing/save", response_model=BreathingAssessmentResponse, tags=["Breathing Assessment"])
async def save_breathing_assessment(req: BreathingAssessmentRequest):
    """
    Saves a completed breathing exercise or spirometry/PFT assessment into the persistent database.
    """
    now = datetime.now()
    created_at = now.isoformat()

    if req.assessment_type == "spirometry_pft":
        # Clinical Spirometry Assessment
        fev1 = req.fev1 or 0.0
        fvc = req.fvc or 0.0
        ratio = req.ratio if req.ratio is not None else ((fev1 / fvc * 100.0) if fvc > 0 else 0.0)
        pef = req.pef or 0.0

        if ratio >= 70.0:
            result_title = "Normal Spirometry Pattern"
            interpretation = (
                f"Preserved FEV1/FVC ratio ({round(ratio, 1)}%). No obstructive ventilatory defect identified "
                "under GOLD spirometry criteria."
            )
            next_steps = "Maintain active cardiovascular exercise and smoke-free lifestyle. Periodic follow-up as indicated."
        else:
            result_title = "Obstructive Ventilatory Pattern"
            interpretation = (
                f"Reduced FEV1/FVC ratio ({round(ratio, 1)}% < 70% threshold). Consistent with airflow obstruction "
                "(characteristic of COPD or Asthma)."
            )
            next_steps = "Clinical correlation by a pulmonologist, pre/post bronchodilator spirometry, and symptom assessment."

        cancer_status = "N/A"
        cancer_stage = "N/A"
        abnormality_desc = f"Spirometry Values: FEV1={fev1}L, FVC={fvc}L, Ratio={round(ratio, 1)}%, PEF={pef}L/min"
        scan_summary = f"{result_title}: {interpretation}"
        clinical_values = {"fev1": fev1, "fvc": fvc, "ratio": round(ratio, 1), "pef": pef}

    else:
        # Guided Breathing Exercise & Capacity Wellness Check
        cycles = req.completed_cycles or 0
        hold_sec = req.breath_hold_sec or 0.0
        duration = req.duration_sec or 0

        result_title = "Guided Breathing Exercise Completed"
        interpretation = (
            f"Successfully executed {cycles} paced breathing cycles ({duration}s duration). "
            f"Maximum sustained breath-hold duration: {hold_sec} seconds. Note: Guided respiratory exercises support "
            "relaxation and thoracic expansion; they do not replace diagnostic clinical spirometry."
        )
        cancer_status = "N/A"
        cancer_stage = "N/A"
        abnormality_desc = f"Wellness Breathing Exercise: {cycles} cycles completed, {hold_sec}s breath-hold."
        scan_summary = interpretation
        next_steps = "Practice daily diaphragmatic deep breathing to enhance respiratory muscle strength and stress reduction."
        clinical_values = {"completed_cycles": cycles, "duration_sec": duration, "breath_hold_sec": hold_sec}

    test_id = save_test_record(
        test_type="Breathing Test",
        detection_result=result_title,
        cancer_status=cancer_status,
        cancer_stage=cancer_stage,
        confidence=1.0,
        image_url="assets/lung_hero_artistic.jpg",
        abnormality_description=abnormality_desc,
        scan_summary=scan_summary,
        recommended_next_steps=next_steps,
        raw_metadata={
            "assessment_type": req.assessment_type,
            "clinical_values": clinical_values,
            "notes": req.notes or ""
        }
    )

    return BreathingAssessmentResponse(
        success=True,
        test_id=test_id,
        created_at=created_at,
        test_type="Breathing Test",
        assessment_type=req.assessment_type,
        result_title=result_title,
        interpretation=interpretation,
        clinical_values=clinical_values,
        disclaimer=MEDICAL_DISCLAIMER
    )


# ==========================================
# Dashboard & Testing History APIs
# ==========================================
@app.get("/api/v1/dashboard/stats", response_model=DashboardStatsResponse, tags=["Dashboard"])
async def fetch_dashboard_stats():
    """
    Returns live summary statistics from actual stored database records.
    """
    stats = get_dashboard_stats()
    return DashboardStatsResponse(**stats)


@app.get("/api/v1/tests", tags=["Dashboard"])
async def fetch_test_history(test_type: str = "all", search: str = "", limit: int = 100):
    """
    Lists testing history records with optional type filtering and search.
    """
    records = list_test_records(test_type=test_type, search=search, limit=limit)
    return records


@app.get("/api/v1/tests/{test_id}", tags=["Dashboard"])
async def fetch_test_details(test_id: str):
    """
    Retrieves full medical report details for a specific test record.
    """
    record = get_test_record_by_id(test_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Test record with ID '{test_id}' not found."
        )
    return record


@app.delete("/api/v1/tests/{test_id}", tags=["Dashboard"])
async def remove_test_record(test_id: str):
    """
    Deletes a specific test record with confirmation.
    """
    success = delete_test_record(test_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Test record with ID '{test_id}' not found or already deleted."
        )
    return {"success": True, "message": f"Test record '{test_id}' deleted successfully."}


# ==========================================
# Page Routing & Static Files
# ==========================================
# Mount static assets
if ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")

# Mount CSS and JS directories directly so relative paths in HTML resolve perfectly
if (FRONTEND_DIR / "css").exists():
    app.mount("/css", StaticFiles(directory=str(FRONTEND_DIR / "css")), name="frontend_css")

if (FRONTEND_DIR / "js").exists():
    app.mount("/js", StaticFiles(directory=str(FRONTEND_DIR / "js")), name="frontend_js")

# Mount frontend root static files
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend_static")

    @app.get("/favicon.ico", include_in_schema=False)
    async def serve_favicon():
        fav = FRONTEND_DIR / "favicon.ico"
        if fav.exists():
            return FileResponse(str(fav), media_type="image/x-icon")
        return JSONResponse({"status": "no favicon"}, status_code=204)

    @app.get("/.well-known/appspecific/{rest_of_path:path}", include_in_schema=False)
    async def serve_devtools_probe(rest_of_path: str):
        return JSONResponse({})

    @app.get("/", include_in_schema=False)
    @app.get("/index.html", include_in_schema=False)
    @app.get("/home", include_in_schema=False)
    async def serve_index():
        index_file = FRONTEND_DIR / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        return JSONResponse({"message": "LungGuard AI API is operational. Visit /docs for OpenAPI documentation."})

    @app.get("/cancer-testing", include_in_schema=False)
    @app.get("/cancer-testing.html", include_in_schema=False)
    async def serve_cancer_testing():
        page = FRONTEND_DIR / "cancer-testing.html"
        if page.exists():
            return FileResponse(str(page))
        return FileResponse(str(FRONTEND_DIR / "index.html"))

    @app.get("/breathing-test", include_in_schema=False)
    @app.get("/breathing-test.html", include_in_schema=False)
    async def serve_breathing_test():
        page = FRONTEND_DIR / "breathing-test.html"
        if page.exists():
            return FileResponse(str(page))
        return FileResponse(str(FRONTEND_DIR / "index.html"))

    @app.get("/dashboard", include_in_schema=False)
    @app.get("/dashboard.html", include_in_schema=False)
    async def serve_dashboard():
        page = FRONTEND_DIR / "dashboard.html"
        if page.exists():
            return FileResponse(str(page))
        return FileResponse(str(FRONTEND_DIR / "index.html"))
