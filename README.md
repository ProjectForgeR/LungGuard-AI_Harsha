# LungGuard AI — AI-Powered Lung CT Image Analysis

> **Medical & Educational Research Disclaimer**  
> *LungGuard AI is an academic AI research prototype and educational demonstration tool, NOT a medical diagnostic device. Its predictions are not a medical diagnosis. Never use this system for clinical triage, confirming, or ruling out pulmonary malignancies. Always consult a certified radiologist or qualified physician for clinical diagnostic evaluation.*

---

## 1. Project Overview

LungGuard AI is a medical-grade AI system that classifies axial chest CT images into three distinct diagnostic categories:
- **Normal**: Intact bronchovascular branching and clear parenchymal attenuation.
- **Benign**: Well-circumscribed, non-invasive pulmonary nodules (e.g. hamartomas, granulomas).
- **Malignant**: Invasive neoplastic lesions characterized by irregular, spiculated margins.

The application combines a modern web frontend with a high-performance Python FastAPI backend powering a fine-tuned **EfficientNetB0** convolutional neural network trained on the benchmark **IQ-OTH-NCCD** dataset.

---

## 2. Research & Model Verification

All architecture parameters and evaluation benchmarks were verified directly against the Colab training notebook (`LungGurard_AI(2).ipynb`):

| Attribute | Verified Value |
|---|---|
| **Base Architecture** | EfficientNetB0 (Pretrained on ImageNet) |
| **Input Shape** | 224 × 224 × 3 (RGB) |
| **Output Classes** | `['Benign', 'Malignant', 'Normal']` (Indices 0, 1, 2) |
| **Output Activation** | Softmax |
| **Loss Function** | Categorical Crossentropy |
| **Optimizer** | Adam (Learning rate = 0.001) |
| **Total Parameters** | 4,053,414 (15.46 MB) |
| **Trainable Parameters** | 3,843 (15.01 KB top classification head) |
| **Training Dataset** | IQ-OTH-NCCD (1,098 CT images from 110 patients) |
| **Dataset Partition** | Train: 767 | Validation: 164 | Test: 166 |

### Evaluation Comparison

- **Original Best Model (`lungguard_best.keras` → `LungGuard_AI_Final.keras`)**:
  - **Test Accuracy**: **94.58%** (157 / 166 correct)
  - **Test Loss**: 0.2184
  - **Confusion Matrix**:
    - Benign (18): 16 Benign, 0 Malignant, 2 Normal
    - Malignant (85): 1 Benign, 83 Malignant, 1 Normal
    - Normal (63): 3 Benign, 2 Malignant, 58 Normal
  - **Per-Class Metrics**:
    - Benign: Precision 0.8000 | Recall 0.8889 | F1 0.8421
    - Malignant: Precision 0.9765 | Recall 0.9765 | F1 0.9765
    - Normal: Precision 0.9508 | Recall 0.9206 | F1 0.9355
- **Fine-Tuning Experiment (`lungguard_finetuned_best.keras`)**:
  - **Test Accuracy**: 68.67%
  - **Test Loss**: 0.6528
  - **Decision**: In Cell 44 of the notebook, `lungguard_best.keras` was copied to `LungGuard_AI_Final.keras` because geometric augmentations in the fine-tuning experiment reduced generalization on this dataset.

### Preprocessing Protocol
In accordance with Keras's `image_dataset_from_directory`, images are loaded in `[0, 255]` range. EfficientNetB0 includes internal normalization layers; therefore, dividing by 255 before feeding to the model is avoided.

---

## 3. Project Directory Structure

```
Harsha_LUNG_AI/
├── assets/
│   ├── educational/                 # Educational anatomy & disease illustrations
│   └── sample_scans/                # Authentic IQ-OTH-NCCD benchmark CT scans
│       ├── benign/
│       ├── malignant/
│       ├── normal/
│       └── metadata.json            # Benchmark catalog
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── config.py                # Configuration and environment settings
│   │   ├── schemas.py               # Pydantic request/response schemas
│   │   ├── preprocessing.py         # Magic byte validation & Lanczos 224x224 tensor formatting
│   │   ├── model_loader.py          # Singleton Keras model manager
│   │   ├── prediction.py            # Prediction pipeline & probability mapping
│   │   └── main.py                  # FastAPI application with CORS & static routes
│   └── requirements.txt             # Python dependencies
├── frontend/
│   ├── index.html                   # Premium responsive web interface
│   ├── css/
│   │   └── style.css                # Custom medical-grade styling & animations
│   ├── js/
│   │   └── app.js                   # Frontend client with live API integration
│   └── assets/                      # Bundled CT scans & illustrations
├── models/
│   ├── LungGuard_AI_Final.keras     # Trained model binary (see instructions below)
│   ├── model_metadata.json          # Complete verified model metadata & test metrics
│   └── verify_model.py              # CLI verification script
├── notebooks/
│   └── LungGurard_AI.ipynb          # Source of truth training notebook
├── tests/
│   ├── test_api.py                  # API endpoint tests
│   ├── test_preprocessing.py        # Image tensor transformation tests
│   └── test_model_verification.py   # Ground-truth verification tests
├── .env.example                     # Environment template
├── .gitignore
├── Dockerfile                       # Production container definition
├── docker-compose.yml               # Container orchestration
├── DEPLOYMENT.md                    # Cloud and VPS deployment guide
└── README.md
```

---

## 4. Setup & Running Instructions

### Step 1: Install Dependencies
```bash
pip install -r backend/requirements.txt
# To run inference with TensorFlow:
pip install tensorflow>=2.15.0
```

### Step 2: Model File Placement
The trained model was saved during Colab training to Google Drive:
- **Source**: `/content/drive/MyDrive/LungGuard_AI/Models/LungGuard_AI_Final.keras`
- **Destination**: Place the `.keras` file in `models/LungGuard_AI_Final.keras`.

Verify placement:
```bash
python models/verify_model.py
```

### Step 3: Run the Test Suite
```bash
python -m pytest tests -v
```

### Step 4: Launch Backend & Frontend
Start the FastAPI server:
```bash
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

Open your browser at:
- **Application**: [http://localhost:8000/](http://localhost:8000/)
- **API Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

---

## 5. API Endpoints

- `GET /health`: Health check and model loading status.
- `GET /api/v1/model-info`: Full verified architecture and evaluation parameters.
- `GET /api/v1/samples`: Authentic IQ-OTH-NCCD benchmark sample catalog.
- `POST /predict`: Upload a lung CT image (JPEG/PNG, <= 10MB) for inference.
- `POST /api/v1/predict-sample/{sample_id}`: Run inference on a benchmark scan.

---

## 6. Known Validation Limitations

1. **2D Image-Level Partition**: Dataset split was performed at the slice level without verified patient ID isolation. Potential intra-patient correlation cannot be excluded.
2. **Single Center Cohort**: IQ-OTH-NCCD consists of axial slices from 110 patients; demographic and institutional diversity is constrained.
3. **Statistical Confidence vs Medical Certainty**: Softmax confidence scores indicate feature alignment with the training distribution, not clinical probability.
