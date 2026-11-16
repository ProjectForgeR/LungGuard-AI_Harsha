# LungGuard AI — Production Deployment Guide

This guide details instructions for deploying the **LungGuard AI** application to local, containerized, and cloud production environments.

---

## 1. Architecture Overview

- **Backend**: Python 3.11 + FastAPI + Uvicorn + TensorFlow/Keras
- **Frontend**: Clean Semantic HTML5 + Custom Modern CSS + Vanilla JS (No build step required, served directly by FastAPI or Nginx/Vercel/Netlify)
- **Model**: EfficientNetB0 (`LungGuard_AI_Final.keras`, ~16.3 MB)
- **Dataset**: IQ-OTH-NCCD (Iraq Oncology Teaching Hospital / National Center for Cancer Diseases)

---

## 2. Model File Placement

The model file was trained in Google Colab and saved to Google Drive:
- **Colab Save Location**: `/content/drive/MyDrive/LungGuard_AI/Models/LungGuard_AI_Final.keras`
- **Source Checkpoint**: `lungguard_best.keras` (copied to `LungGuard_AI_Final.keras` in Cell 44 with 94.58% test accuracy)

### Local Placement
Place the file into:
```
Harsha_LUNG_AI/
  └── models/
        └── LungGuard_AI_Final.keras
```

Verify placement by running:
```bash
python models/verify_model.py
```

---

## 3. Local Development & Testing

### Step 1: Install Dependencies
```bash
pip install -r backend/requirements.txt
# To run inference with TensorFlow:
pip install tensorflow>=2.15.0
```

### Step 2: Run Unit Tests
```bash
python -m pytest tests -v
```

### Step 3: Start FastAPI Application
From the project root:
```bash
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

- **Website**: `http://localhost:8000/`
- **Interactive API Documentation (Swagger)**: `http://localhost:8000/docs`
- **System Health Check**: `http://localhost:8000/health`
- **Model Metadata**: `http://localhost:8000/api/v1/model-info`

---

## 4. Containerized Deployment (Docker)

### Using Docker Compose
```bash
# 1. Ensure models/LungGuard_AI_Final.keras is in the models/ directory
# 2. Build and start container
docker-compose up --build -d

# 3. Check logs
docker-compose logs -f
```

### Direct Docker Build
```bash
docker build -t lungguard-ai:latest .
docker run -d -p 8000:8000 -v $(pwd)/models:/app/models --name lungguard lungguard-ai:latest
```

---

## 5. Cloud Hosting Options

### Option A: Render / Railway (Unified Backend + Frontend)
1. Push the repository to GitHub (ensure `.gitignore` excludes large model binaries if over GitHub limits, or use Git LFS).
2. Set Environment Variables:
   - `PORT=8000`
   - `MODEL_PATH=models/LungGuard_AI_Final.keras`
   - `CORS_ORIGINS=*`
3. Provide the model file via:
   - Git LFS (`git lfs track "models/*.keras"`)
   - An S3/GCS bucket download in your startup script (`curl -o models/LungGuard_AI_Final.keras <S3_SIGNED_URL>`)
4. Build Command:
   ```bash
   pip install -r backend/requirements.txt && pip install tensorflow-cpu>=2.15.0
   ```
5. Start Command:
   ```bash
   uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT --workers 2
   ```

### Option B: Decoupled Deployment (Frontend on Vercel/Netlify + Backend on Cloud Run/Render)
1. **Frontend**:
   - Deploy `frontend/` to Vercel or Netlify.
   - In `frontend/js/app.js`, set `API_BASE` to your deployed backend URL:
     ```javascript
     const API_BASE = 'https://your-backend-api.onrender.com';
     ```
2. **Backend**:
   - Deploy FastAPI container to Google Cloud Run, AWS ECS, or Render.
   - Configure `CORS_ORIGINS=https://your-frontend.vercel.app`.

---

## 6. Security & Operational Hardening

- **File Header Magic Byte Validation**: Inspects JPEG (`\xFF\xD8\xFF`) and PNG (`\x89PNG`) signatures to prevent MIME spoofing.
- **Payload Limits**: 10 MB strict file size ceiling.
- **In-Memory Buffer Processing**: Uploaded images are preprocessed in transient RAM streams and never permanently stored on the server filesystem.
- **Medical Disclaimer**: Persistently rendered across all API prediction outputs and frontend interfaces.
