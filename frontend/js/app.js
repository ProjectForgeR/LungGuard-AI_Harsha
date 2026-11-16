/**
 * LungGuard AI — Frontend Application Logic
 * Integrates directly with FastAPI backend endpoints:
 * - GET /health
 * - GET /api/v1/model-info
 * - GET /api/v1/samples
 * - POST /predict
 * - POST /api/v1/predict-sample/{sample_id}
 */

// Determine API Base URL (same origin if served together, or localhost:8000)
const API_BASE = window.location.origin.includes(':8000') || window.location.origin.includes('localhost')
  ? window.location.origin
  : 'http://localhost:8000';

// State Management
const state = {
  currentFile: null,
  previewUrl: null,
  isAnalyzing: false,
  modelInfo: null,
  samples: [],
  backendOnline: false,
  modelLoaded: false,
  theme: localStorage.getItem('lungguard_theme') || 'light'
};

// DOM Elements
const elements = {
  // Theme Toggle
  themeToggleBtn: document.getElementById('theme-toggle-btn'),

  // System Status
  statusPill: document.getElementById('system-status-pill'),
  statusText: document.getElementById('system-status-text'),
  statusDot: document.getElementById('system-status-dot'),
  
  // Navigation
  mobileMenuBtn: document.getElementById('mobile-menu-btn'),
  navMenu: document.querySelector('.nav-menu'),
  navLinks: document.getElementById('nav-links'),

  // Dropzone & Upload
  dropzone: document.getElementById('dropzone'),
  fileInput: document.getElementById('file-input'),
  previewContainer: document.getElementById('preview-container'),
  previewImage: document.getElementById('preview-image'),
  removePreviewBtn: document.getElementById('remove-preview-btn'),
  scanningOverlay: document.getElementById('scanning-overlay'),
  analyzeBtn: document.getElementById('analyze-btn'),
  benchmarkChipsContainer: document.getElementById('benchmark-chips'),

  // Results
  resultsPlaceholder: document.getElementById('results-placeholder'),
  activeResults: document.getElementById('active-results'),
  errorAlert: document.getElementById('error-alert'),
  errorText: document.getElementById('error-text'),
  predClassText: document.getElementById('pred-class-text'),
  confidenceValue: document.getElementById('confidence-value'),
  probNormalFill: document.getElementById('prob-normal-fill'),
  probNormalVal: document.getElementById('prob-normal-val'),
  probBenignFill: document.getElementById('prob-benign-fill'),
  probBenignVal: document.getElementById('prob-benign-val'),
  probMalignantFill: document.getElementById('prob-malignant-fill'),
  probMalignantVal: document.getElementById('prob-malignant-val'),
  inferenceTimeVal: document.getElementById('inference-time-val'),
  imageDimVal: document.getElementById('image-dim-val'),
  resetAnalysisBtn: document.getElementById('reset-analysis-btn'),

  // Clinical Report Fields
  cancerStatusVal: document.getElementById('cancer-status-val'),
  stageRow: document.getElementById('stage-classification-row'),
  cancerStageVal: document.getElementById('cancer-stage-val'),
  cancerStageNote: document.getElementById('cancer-stage-note'),
  abnormalityDescVal: document.getElementById('abnormality-desc-val'),
  scanSummaryVal: document.getElementById('scan-summary-val'),
  nextStepsVal: document.getElementById('next-steps-val'),

  // Gallery
  galleryGrid: document.getElementById('gallery-grid'),
  filterBtns: document.querySelectorAll('.filter-btn'),

  // Telemetry
  telemetryStatus: document.getElementById('telemetry-status'),
  telemetryDevice: document.getElementById('telemetry-device'),
  telemetryModel: document.getElementById('telemetry-model'),
  telemetryTimestamp: document.getElementById('telemetry-timestamp'),

  // Modal
  modal: document.getElementById('image-modal'),
  modalCloseBtn: document.getElementById('modal-close-btn'),
  modalImg: document.getElementById('modal-img'),
  modalTitle: document.getElementById('modal-title'),
  modalDesc: document.getElementById('modal-desc'),
  modalClass: document.getElementById('modal-class'),

  // Model Setup Modal
  modelSetupModal: document.getElementById('model-setup-modal'),
  modelModalCloseBtn: document.getElementById('model-modal-close-btn'),
  btnRecheckModel: document.getElementById('btn-recheck-model')
};

// ==========================================
// Initializer
// ==========================================
document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  initSystemHealthCheck();
  initModelMetadata();
  initSampleScans();
  setupEventListeners();
  initAnatomyCarousel();
});

// ==========================================
// Theme Management (Clean Light / Dark)
// ==========================================
function initTheme() {
  document.documentElement.setAttribute('data-theme', state.theme);
  updateThemeButtonUI();

  if (elements.themeToggleBtn) {
    elements.themeToggleBtn.addEventListener('click', () => {
      state.theme = state.theme === 'light' ? 'dark' : 'light';
      localStorage.setItem('lungguard_theme', state.theme);
      document.documentElement.setAttribute('data-theme', state.theme);
      updateThemeButtonUI();
    });
  }
}

function updateThemeButtonUI() {
  if (!elements.themeToggleBtn) return;
  if (state.theme === 'dark') {
    elements.themeToggleBtn.innerHTML = '🌙 Dark';
  } else {
    elements.themeToggleBtn.innerHTML = '☀️ Light';
  }
}

// ==========================================
// Backend Health Check & Telemetry
// ==========================================
async function initSystemHealthCheck() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (res.ok) {
      const data = await res.json();
      state.backendOnline = true;
      state.modelLoaded = data.model_loaded;

      if (elements.statusText) {
        if (data.model_loaded) {
          elements.statusText.textContent = 'System Ready';
        } else {
          elements.statusText.textContent = 'Setup Model';
        }
      }
      if (elements.statusDot) {
        elements.statusDot.className = data.model_loaded ? 'status-dot' : 'status-dot degraded';
      }

      // Update telemetry display
      if (elements.telemetryStatus) {
        elements.telemetryStatus.textContent = data.status.toUpperCase();
        elements.telemetryStatus.className = `table-badge ${data.status === 'healthy' ? 'verified' : 'experimental'}`;
      }
      if (elements.telemetryDevice) elements.telemetryDevice.textContent = data.device;
      if (elements.telemetryModel) elements.telemetryModel.textContent = data.model_name || 'EfficientNetB0 (Unloaded)';
      if (elements.telemetryTimestamp) elements.telemetryTimestamp.textContent = new Date(data.timestamp).toLocaleTimeString();

    } else {
      throw new Error(`HTTP ${res.status}`);
    }
  } catch (err) {
    state.backendOnline = false;
    if (elements.statusText) elements.statusText.textContent = 'Offline';
    if (elements.statusDot) elements.statusDot.className = 'status-dot degraded';
    console.warn('Backend connection notice:', err);
  }
}

// ==========================================
// Model Metadata & Benchmarks
// ==========================================
async function initModelMetadata() {
  try {
    const res = await fetch(`${API_BASE}/api/v1/model-info`);
    if (res.ok) {
      state.modelInfo = await res.json();
    }
  } catch (err) {
    console.log('Using bundled verification metrics.');
  }
}

// ==========================================
// Benchmark Samples Loader
// ==========================================
async function initSampleScans() {
  try {
    const res = await fetch(`${API_BASE}/api/v1/samples`);
    if (res.ok) {
      state.samples = await res.json();
      renderBenchmarkChips(state.samples);
      renderGalleryCards(state.samples);
    } else {
      loadFallbackSamples();
    }
  } catch (err) {
    loadFallbackSamples();
  }

  // Check URL query param for sample auto-load (e.g. redirected from landing gallery)
  try {
    const urlParams = new URLSearchParams(window.location.search);
    const sampleParam = urlParams.get('sample');
    if (sampleParam && state.samples && state.samples.length > 0) {
      const match = state.samples.find(s => s.id === sampleParam);
      if (match) {
        setTimeout(() => loadSampleIntoStudio(match), 300);
      }
    }
  } catch (_) {}
}

function loadFallbackSamples() {
  const fallback = [
    {
      id: 'benign-1',
      title: 'IQ-OTH Benign Case 01',
      class_label: 'Benign',
      description: 'Well-circumscribed nodular density with smooth margins and non-invasive architectural profile.',
      image_url: 'assets/sample_benign_bengin_case_(1).jpg'
    },
    {
      id: 'malignant-1',
      title: 'IQ-OTH Malignant Case 01',
      class_label: 'Malignant',
      description: 'Irregular soft-tissue lesion displaying spiculated architectural margins and parenchymal invasion.',
      image_url: 'assets/sample_malignant_malignant_case_(1).jpg'
    },
    {
      id: 'normal-1',
      title: 'IQ-OTH Normal Case 01',
      class_label: 'Normal',
      description: 'Clear lung parenchymal attenuation, symmetric bronchovascular branching without focal opacities.',
      image_url: 'assets/sample_normal_normal_case_(1).jpg'
    }
  ];
  state.samples = fallback;
  renderBenchmarkChips(fallback);
  renderGalleryCards(fallback);
}

function renderBenchmarkChips(samples) {
  if (!elements.benchmarkChipsContainer) return;
  elements.benchmarkChipsContainer.innerHTML = '';

  samples.slice(0, 4).forEach(sample => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = `chip-btn chip-${sample.class_label.toLowerCase()}`;
    btn.innerHTML = `
      <span>${sample.class_label === 'Normal' ? '🟢' : sample.class_label === 'Benign' ? '🟡' : '🔴'}</span>
      <span>${sample.title}</span>
    `;
    btn.addEventListener('click', () => loadSampleIntoStudio(sample));
    elements.benchmarkChipsContainer.appendChild(btn);
  });
}

function renderGalleryCards(samples) {
  if (!elements.galleryGrid) return;
  elements.galleryGrid.innerHTML = '';

  samples.forEach(sample => {
    const card = document.createElement('div');
    card.className = 'gallery-card';
    card.dataset.category = sample.class_label.toLowerCase();

    card.innerHTML = `
      <div class="gallery-img-wrap" onclick="openGalleryModal('${sample.id}')">
        <img src="${sample.image_url}" alt="${sample.title}" />
        <div class="gallery-overlay">
          <button class="zoom-btn">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7"/></svg>
            Inspect CT
          </button>
        </div>
      </div>
      <div class="gallery-info">
        <div class="gallery-top">
          <h4>${sample.title}</h4>
          <span class="class-pill ${sample.class_label.toLowerCase()}">${sample.class_label}</span>
        </div>
        <p>${sample.description}</p>
        <button class="btn-test-sample" onclick="loadSampleFromId('${sample.id}')">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
          Load in AI Studio
        </button>
      </div>
    `;
    elements.galleryGrid.appendChild(card);
  });
}

// ==========================================
// Event Listeners
// ==========================================
function setupEventListeners() {
  // Mobile Nav Toggle
  if (elements.mobileMenuBtn && elements.navMenu) {
    elements.mobileMenuBtn.addEventListener('click', () => {
      elements.navMenu.classList.toggle('active');
    });

    const navLinksList = document.querySelectorAll('.nav-link');
    navLinksList.forEach(link => {
      link.addEventListener('click', () => {
        elements.navMenu.classList.remove('active');
      });
    });
  }

  // Dropzone drag-and-drop
  if (elements.dropzone) {
    ['dragenter', 'dragover'].forEach(eventName => {
      elements.dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        elements.dropzone.classList.add('drag-active');
      });
    });

    ['dragleave', 'drop'].forEach(eventName => {
      elements.dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        elements.dropzone.classList.remove('drag-active');
      });
    });

    elements.dropzone.addEventListener('drop', (e) => {
      const files = e.dataTransfer.files;
      if (files && files.length > 0) {
        handleFileSelection(files[0]);
      }
    });

    elements.dropzone.addEventListener('click', () => {
      elements.fileInput.click();
    });
  }

  // File Input Change
  if (elements.fileInput) {
    elements.fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files.length > 0) {
        handleFileSelection(e.target.files[0]);
      }
    });
  }

  // Remove preview
  if (elements.removePreviewBtn) {
    elements.removePreviewBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      resetStudio();
    });
  }

  // Analyze button click
  if (elements.analyzeBtn) {
    elements.analyzeBtn.addEventListener('click', executeAnalysis);
  }

  // Reset analysis
  if (elements.resetAnalysisBtn) {
    elements.resetAnalysisBtn.addEventListener('click', resetStudio);
  }

  // Gallery Filters
  if (elements.filterBtns) {
    elements.filterBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        elements.filterBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const filter = btn.dataset.filter;
        filterGallery(filter);
      });
    });
  }

  // Modal Close
  if (elements.modalCloseBtn) {
    elements.modalCloseBtn.addEventListener('click', closeGalleryModal);
  }
  if (elements.modal) {
    elements.modal.addEventListener('click', (e) => {
      if (e.target === elements.modal) closeGalleryModal();
    });
  }

  // Status Pill Click
  if (elements.statusPill) {
    elements.statusPill.style.cursor = 'pointer';
    elements.statusPill.addEventListener('click', () => {
      if (!state.modelLoaded && elements.modelSetupModal) {
        elements.modelSetupModal.style.display = 'flex';
      }
    });
  }

  // Model Setup Modal Listeners
  if (elements.modelModalCloseBtn && elements.modelSetupModal) {
    elements.modelModalCloseBtn.addEventListener('click', () => {
      elements.modelSetupModal.style.display = 'none';
    });
  }
  if (elements.modelSetupModal) {
    elements.modelSetupModal.addEventListener('click', (e) => {
      if (e.target === elements.modelSetupModal) {
        elements.modelSetupModal.style.display = 'none';
      }
    });
  }
  if (elements.btnRecheckModel) {
    elements.btnRecheckModel.addEventListener('click', async () => {
      elements.btnRecheckModel.textContent = 'Checking...';
      await initSystemHealthCheck();
      if (state.modelLoaded) {
        elements.btnRecheckModel.textContent = 'Connected Successfully!';
        setTimeout(() => {
          if (elements.modelSetupModal) elements.modelSetupModal.style.display = 'none';
          hideError();
        }, 800);
      } else {
        elements.btnRecheckModel.textContent = 'Not Found Yet - Try Again';
        setTimeout(() => {
          elements.btnRecheckModel.textContent = 'Check Again & Connect';
        }, 2000);
      }
    });
  }
}

// ==========================================
// File Validation & Preview
// ==========================================
function handleFileSelection(file) {
  hideError();

  const validTypes = ['image/jpeg', 'image/png', 'image/jpg', 'application/dicom'];
  const ext = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
  const validExts = ['.jpg', '.jpeg', '.png', '.dcm'];

  if (!validTypes.includes(file.type) && !validExts.includes(ext)) {
    showError('Unsupported file type. Please upload a valid Lung CT scan (.jpg, .jpeg, .png, .dcm).');
    return;
  }

  if (file.size > 10 * 1024 * 1024) {
    showError('File exceeds maximum size of 10 MB. Please select an optimized lung CT scan image.');
    return;
  }

  state.currentFile = file;

  if (ext === '.dcm') {
    elements.previewImage.src = 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="300" height="300" viewBox="0 0 24 24" fill="none" stroke="%230284c7" stroke-width="1.5"><rect width="18" height="18" x="3" y="3" rx="2"/><path d="M7 10h4a2 2 0 0 1 2 2v0a2 2 0 0 1-2 2H7z"/><path d="M17 14V10"/><path d="M14 14V10"/></svg>';
    elements.dropzone.style.display = 'none';
    elements.previewContainer.style.display = 'block';
    elements.analyzeBtn.disabled = false;
  } else {
    const reader = new FileReader();
    reader.onload = (e) => {
      state.previewUrl = e.target.result;
      elements.previewImage.src = e.target.result;
      elements.dropzone.style.display = 'none';
      elements.previewContainer.style.display = 'block';
      elements.analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
  }
}

async function loadSampleIntoStudio(sample) {
  try {
    hideError();
    const studioEl = document.getElementById('analysis-studio');
    if (!studioEl) {
      window.location.href = `cancer-testing.html?sample=${encodeURIComponent(sample.id || '')}`;
      return;
    }
    const res = await fetch(sample.image_url);
    const blob = await res.blob();
    const file = new File([blob], sample.filename || `${sample.id}.jpg`, { type: 'image/jpeg' });
    handleFileSelection(file);

    studioEl.scrollIntoView({ behavior: 'smooth' });
  } catch (err) {
    showError('Unable to load benchmark sample image: ' + err.message);
  }
}

window.loadSampleFromId = (sampleId) => {
  const sample = state.samples.find(s => s.id === sampleId);
  if (sample) {
    loadSampleIntoStudio(sample);
  }
};

// ==========================================
// Execute AI Analysis
// ==========================================
async function executeAnalysis() {
  if (!state.currentFile || state.isAnalyzing) return;

  state.isAnalyzing = true;
  elements.analyzeBtn.disabled = true;
  elements.scanningOverlay.style.display = 'flex';
  hideError();

  const formData = new FormData();
  formData.append('file', state.currentFile);

  try {
    const response = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      let detailMsg = `HTTP Error ${response.status}`;
      try {
        const errorJson = await response.json();
        detailMsg = errorJson.detail || detailMsg;
      } catch (_) {}

      if (response.status === 503) {
        if (elements.modelSetupModal) {
          elements.modelSetupModal.style.display = 'flex';
        }
        throw new Error("AI Model File Missing: Please download 'LungGuard_AI_Final.keras' from Google Drive and place it in 'Harsha_LUNG_AI/models/'.");
      } else {
        throw new Error(detailMsg);
      }
    }

    const result = await response.json();
    renderPredictionResults(result);

  } catch (error) {
    console.error('Prediction request failed:', error);
    showError(error.message || 'Network error occurred while contacting AI analysis backend.');
    elements.activeResults.style.display = 'none';
    elements.resultsPlaceholder.style.display = 'flex';
  } finally {
    state.isAnalyzing = false;
    elements.analyzeBtn.disabled = false;
    elements.scanningOverlay.style.display = 'none';
  }
}

// ==========================================
// Render Prediction Results
// ==========================================
function renderPredictionResults(res) {
  elements.resultsPlaceholder.style.display = 'none';
  elements.activeResults.style.display = 'flex';

  const detectionResult = res.detection_result || res.predicted_class;
  const confidencePercent = (res.confidence * 100).toFixed(1);

  const testBadge = document.getElementById('test-id-badge');
  if (testBadge && res.test_id) {
    testBadge.textContent = `Record ID: ${res.test_id}`;
  }

  elements.predClassText.textContent = detectionResult;
  
  if (res.is_cancer_detected || res.raw_class === 'Malignant' || (detectionResult.includes('Detected') && !detectionResult.includes('No'))) {
    elements.predClassText.className = 'pred-class-large malignant';
  } else if (detectionResult.includes('Inconclusive')) {
    elements.predClassText.className = 'pred-class-large inconclusive';
  } else {
    elements.predClassText.className = 'pred-class-large normal';
  }

  elements.confidenceValue.textContent = `${confidencePercent}%`;

  // Cancer Status
  if (elements.cancerStatusVal) {
    elements.cancerStatusVal.textContent = res.cancer_status || (res.is_cancer_detected ? 'Positive for Malignant Lesion' : 'Negative for Malignancy');
    elements.cancerStatusVal.style.color = res.is_cancer_detected 
      ? 'var(--class-malignant)' 
      : (detectionResult.includes('Inconclusive') ? 'var(--class-benign)' : 'var(--class-normal)');
  }

  // Cancer Stage Classification (ONLY shown if cancer is detected)
  if (elements.stageRow) {
    if (res.is_cancer_detected && res.predicted_stage) {
      elements.stageRow.style.display = 'flex';
      if (elements.cancerStageVal) elements.cancerStageVal.textContent = res.predicted_stage;
      if (elements.cancerStageNote && res.stage_classification_note) elements.cancerStageNote.textContent = res.stage_classification_note;
    } else {
      elements.stageRow.style.display = 'none';
    }
  }

  // Abnormality Description
  if (elements.abnormalityDescVal) {
    elements.abnormalityDescVal.textContent = res.abnormality_description || 'No focal abnormality detected across parenchymal fields.';
  }

  // Scan Analysis Summary
  if (elements.scanSummaryVal) {
    elements.scanSummaryVal.textContent = res.scan_summary || `Analysis completed with ${confidencePercent}% statistical confidence.`;
  }

  // Recommended Next Steps
  if (elements.nextStepsVal) {
    elements.nextStepsVal.textContent = res.recommended_next_steps || 'Consult a licensed physician or thoracic radiologist for clinical correlation.';
  }

  // Probabilities Bars
  const probs = res.class_probabilities || {};
  const normalProb = ((probs['Normal'] || 0) * 100).toFixed(1);
  const benignProb = ((probs['Benign'] || 0) * 100).toFixed(1);
  const malignantProb = ((probs['Malignant'] || 0) * 100).toFixed(1);

  elements.probNormalVal.textContent = `${normalProb}%`;
  elements.probNormalFill.style.width = `${normalProb}%`;

  elements.probBenignVal.textContent = `${benignProb}%`;
  elements.probBenignFill.style.width = `${benignProb}%`;

  elements.probMalignantVal.textContent = `${malignantProb}%`;
  elements.probMalignantFill.style.width = `${malignantProb}%`;

  elements.inferenceTimeVal.textContent = `${res.inference_time_ms} ms`;
  if (res.image_meta && res.image_meta.original_dimensions) {
    const [w, h] = res.image_meta.original_dimensions;
    elements.imageDimVal.textContent = `${w} × ${h} px (Resized to 224 × 224)`;
  } else {
    elements.imageDimVal.textContent = '224 × 224 × 3';
  }

  if (window.innerWidth < 1024) {
    elements.activeResults.scrollIntoView({ behavior: 'smooth' });
  }
}

// ==========================================
// Gallery Filtering & Lightbox Modal
// ==========================================
function filterGallery(category) {
  const cards = document.querySelectorAll('.gallery-card');
  cards.forEach(card => {
    if (category === 'all' || card.dataset.category === category) {
      card.style.display = 'block';
    } else {
      card.style.display = 'none';
    }
  });
}

window.openGalleryModal = (sampleId) => {
  const sample = state.samples.find(s => s.id === sampleId);
  if (!sample) return;

  elements.modalImg.src = sample.image_url;
  elements.modalTitle.textContent = sample.title;
  elements.modalDesc.textContent = sample.description;
  elements.modalClass.textContent = sample.class_label;
  elements.modalClass.className = `class-pill ${sample.class_label.toLowerCase()}`;
  elements.modal.style.display = 'flex';
};

window.openPosterModal = (imgSrc, title, desc, tag) => {
  if (!elements.modal) return;
  elements.modalImg.src = imgSrc;
  elements.modalTitle.textContent = title;
  elements.modalDesc.textContent = desc;
  elements.modalClass.textContent = tag || 'Clinical Awareness';
  elements.modalClass.className = 'class-pill benign';
  elements.modal.style.display = 'flex';
};

// ==========================================
// Anatomy Showcase Auto-Carousel (2 seconds)
// ==========================================
const anatomySlides = [
  {
    src: 'assets/lungs_anatomy.jpg',
    caption: 'Figure 1: Morphological anatomy of the tracheobronchial tree, lobes, and alveolar micro-architecture.'
  },
  {
    src: 'assets/lung_awareness_3d.jpg',
    caption: 'Figure 2: 3D anatomical model of tracheobronchial arborization and dual pulmonary lobes.'
  },
  {
    src: 'assets/lung_pathology_macro.jpg',
    caption: 'Figure 3: High-resolution macro architecture of bronchial conduits and parenchymal tissue.'
  }
];

let anatomyCurrentIndex = 0;
let anatomyInterval = null;

function initAnatomyCarousel() {
  const img = document.getElementById('anatomy-main-img');
  if (!img) return;

  // Preload slides for smooth instant transitions
  anatomySlides.forEach(slide => {
    const pre = new Image();
    pre.src = slide.src;
  });

  startAnatomyCarousel();
}

function startAnatomyCarousel() {
  if (anatomyInterval) clearInterval(anatomyInterval);
  anatomyInterval = setInterval(() => {
    anatomyCurrentIndex = (anatomyCurrentIndex + 1) % anatomySlides.length;
    showAnatomySlide(anatomyCurrentIndex);
  }, 2000); // changes every 2 seconds
}

function showAnatomySlide(index) {
  const img = document.getElementById('anatomy-main-img');
  const captionEl = document.getElementById('anatomy-caption-text');
  const buttons = document.querySelectorAll('.anatomy-sel-btn');
  const slide = anatomySlides[index];

  if (!slide || !img) return;

  // Smooth fade
  img.style.opacity = '0.35';

  setTimeout(() => {
    img.src = slide.src;
    img.style.opacity = '1';
    if (captionEl) captionEl.textContent = slide.caption;
  }, 180);

  buttons.forEach((b, i) => {
    if (i === index) b.classList.add('active');
    else b.classList.remove('active');
  });
}

window.switchAnatomyView = (imgSrc, caption, btn) => {
  const buttons = Array.from(document.querySelectorAll('.anatomy-sel-btn'));
  const foundIdx = buttons.indexOf(btn);
  if (foundIdx !== -1) {
    anatomyCurrentIndex = foundIdx;
  }
  showAnatomySlide(anatomyCurrentIndex);
  startAnatomyCarousel();
};

function closeGalleryModal() {
  if (elements.modal) elements.modal.style.display = 'none';
}

function resetStudio() {
  state.currentFile = null;
  state.previewUrl = null;
  state.isAnalyzing = false;

  if (elements.fileInput) elements.fileInput.value = '';
  if (elements.previewImage) elements.previewImage.src = '';
  if (elements.previewContainer) elements.previewContainer.style.display = 'none';
  if (elements.dropzone) elements.dropzone.style.display = 'flex';
  if (elements.analyzeBtn) elements.analyzeBtn.disabled = true;

  if (elements.activeResults) elements.activeResults.style.display = 'none';
  if (elements.resultsPlaceholder) elements.resultsPlaceholder.style.display = 'flex';
  if (elements.stageRow) elements.stageRow.style.display = 'none';
  hideError();
}

function showError(msg) {
  if (elements.errorText) elements.errorText.textContent = msg;
  if (elements.errorAlert) elements.errorAlert.style.display = 'flex';
}

function hideError() {
  if (elements.errorAlert) elements.errorAlert.style.display = 'none';
}
