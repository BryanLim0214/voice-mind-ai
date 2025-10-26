<!-- 8fe6a216-4a58-43db-8eec-cf1a873ba774 8593ba19-e440-4b17-9fd5-d8f1a533bd63 -->
# Multi-Disorder Voice Screening Platform

## Project Architecture

**Tech Stack:**

- Feature extraction: openSMILE (eGeMAPS), pyAudioAnalysis
- Models: HuBERT (transformers), scikit-learn (ensemble), PyTorch
- Backend: Flask/FastAPI with audio processing
- Frontend: Modern HTML/CSS/JS with audio recording
- Data: Voiceome (primary), DAIC-WOZ, Vocal Mind datasets

## Implementation Phases

### Phase 1: Project Setup & Data Pipeline

**1.1 Environment Configuration**

- Create `requirements.txt` with core dependencies:
  - `opensmile`, `transformers`, `torch`, `torchaudio`
  - `scikit-learn`, `xgboost`, `pandas`, `numpy`
  - `flask`, `flask-cors`, `librosa`, `soundfile`
  - `pyAudioAnalysis`, `matplotlib`, `seaborn`

**1.2 Dataset Acquisition**

- Download Voiceome dataset from GitHub
- Apply for DAIC-WOZ access (include IRB exemption note for educational research)
- Download Vocal Mind dataset from PhysioNet
- Create unified data structure: `/data/{voiceome,daic,vocal_mind}/`

**1.3 Data Preprocessing Pipeline**

Create `src/data/preprocess.py`:

- Audio normalization (16kHz sampling rate, mono channel)
- Silence removal and voice activity detection
- 60-second segment extraction or padding
- Label mapping for multi-disorder classification

### Phase 2: Feature Engineering

**2.1 Acoustic Feature Extraction**

Create `src/features/extract_features.py`:

```python
import opensmile
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals
)
# Extract 88 clinically validated features
```

**2.2 Statistical Features**

- Prosodic features: pitch variance, speech rate, pause duration
- Voice quality: jitter, shimmer, HNR (harmonic-to-noise ratio)
- Energy features: loudness dynamics, MFCC statistics
- Create feature normalization by age/gender demographics

**Key Files:**

- `src/features/prosodic.py` - Pitch and timing analysis
- `src/features/voice_quality.py` - Acoustic quality metrics
- `src/features/statistical.py` - Aggregate feature computations

### Phase 3: Baseline Models (eGeMAPS + Classical ML)

**3.1 Multi-Task Ensemble Models**

Create `src/models/baseline.py`:

- Separate classifiers for each disorder (depression, anxiety, PTSD, MCI)
- Ensemble: Random Forest, XGBoost, SVM with RBF kernel
- Voting classifier for final predictions
- Cross-validation with stratified k-fold (k=5)

**3.2 Confidence Scoring**

- Probability calibration using Platt scaling
- Uncertainty quantification via prediction variance
- Threshold optimization for clinical sensitivity/specificity

**Evaluation Metrics:**

- AUC-ROC, F1-score, sensitivity, specificity per disorder
- Confusion matrices for multi-class performance
- Clinical validation: compare to PHQ-9/GAD-7 scores

### Phase 4: Deep Learning Model (HuBERT)

**4.1 Transfer Learning Setup**

Create `src/models/hubert_model.py`:

```python
from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor

# Load pre-trained HuBERT base (95M parameters)
model = HubertForSequenceClassification.from_pretrained(
    "facebook/hubert-base-ls960"
)
# Add multi-task heads for 4 disorders
```

**4.2 Multi-Task Learning Architecture**

- Shared HuBERT encoder (frozen for efficiency on local hardware)
- Task-specific classification heads with dropout
- Weighted loss function balancing all disorders
- Fine-tune only classification heads initially

**4.3 Training Strategy (CPU/Local GPU Optimized)**

- Use gradient accumulation (batch_size=4, accumulation_steps=8)
- Mixed precision training (FP16) if GPU available
- Early stopping with patience=5 epochs
- Optional: Use Google Colab free GPU for initial fine-tuning

**Key File:** `src/models/train_hubert.py`

### Phase 5: Model Fusion & Interpretation

**5.1 Ensemble Strategy**

Create `src/models/ensemble.py`:

- Combine eGeMAPS baseline + HuBERT predictions
- Weighted averaging based on validation performance
- Final confidence scores with uncertainty bounds

**5.2 Explainability Module**

Create `src/explainability/interpret.py`:

- SHAP values for feature importance (eGeMAPS)
- Attention visualization for HuBERT
- Clinical report generation with top contributing features
- Deviation scores from age/gender norms

### Phase 6: Web Application

**6.1 Backend API**

Create `app/api.py` (Flask):

```python
@app.route('/analyze', methods=['POST'])
def analyze_voice():
    # Receive audio file
    # Extract features
    # Run ensemble inference
    # Return multi-disorder predictions + confidence + explanations
```

**Endpoints:**

- `POST /analyze` - Upload audio, get predictions
- `GET /health` - System status
- `POST /feedback` - Collect clinical validation feedback

**6.2 Frontend Interface**

Create `app/static/` and `app/templates/`:

- Modern responsive design (Bootstrap/Tailwind CSS)
- 60-second voice recorder with visual feedback
- Real-time waveform visualization
- Results dashboard showing:
  - Risk scores for each disorder (0-100 scale)
  - Confidence intervals
  - Key voice features detected
  - Clinical recommendations (if high risk detected)
- Professional clinical styling for credibility

**6.3 Audio Processing**

- Browser-based audio recording (MediaRecorder API)
- Format conversion to WAV 16kHz mono
- Noise reduction preprocessing
- File size validation

### Phase 7: Evaluation & Validation

**7.1 Clinical Validation**

Create `notebooks/clinical_validation.ipynb`:

- Compare predictions to standardized questionnaires (PHQ-9, GAD-7)
- Correlation analysis with gold-standard diagnoses
- Subgroup analysis (age, gender, severity levels)

**7.2 Performance Benchmarks**

- Generate ROC curves for all disorders
- Create confusion matrices
- Calculate clinical metrics: PPV, NPV, likelihood ratios
- Document model limitations and edge cases

**7.3 Documentation**

Create comprehensive `README.md`:

- Model architecture explanation
- Dataset descriptions and citations
- Installation instructions
- Usage examples with sample audio
- Clinical interpretation guidelines
- Ethical considerations and limitations

## Project Structure

```
DepressionScreening/
├── data/
│   ├── raw/              # Original datasets
│   ├── processed/        # Preprocessed audio + labels
│   └── features/         # Extracted feature matrices
├── src/
│   ├── data/
│   │   ├── preprocess.py
│   │   └── loader.py
│   ├── features/
│   │   ├── extract_features.py
│   │   ├── prosodic.py
│   │   └── voice_quality.py
│   ├── models/
│   │   ├── baseline.py
│   │   ├── hubert_model.py
│   │   ├── train_hubert.py
│   │   └── ensemble.py
│   └── explainability/
│       └── interpret.py
├── app/
│   ├── api.py            # Flask backend
│   ├── static/           # CSS, JS, images
│   └── templates/        # HTML templates
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   ├── model_training.ipynb
│   └── clinical_validation.ipynb
├── models/               # Saved model weights
├── requirements.txt
└── README.md
```

## Key Technical Achievements

1. **Multi-Dataset Integration**: Combines 3 clinical datasets for robustness
2. **Hybrid ML Architecture**: Classical ML + deep learning ensemble
3. **Clinical Interpretability**: eGeMAPS features with SHAP explanations
4. **Multi-Task Learning**: Shared representations for related disorders
5. **Confidence Quantification**: Uncertainty-aware predictions for clinical trust
6. **Real-World Deployment**: Production-ready web application

## Expected Performance

Based on literature:

- Depression detection: AUC 0.75-0.85
- Anxiety detection: AUC 0.70-0.80
- PTSD detection: AUC 0.72-0.82
- Combined screening: 60-second assessment vs. 10-minute questionnaire

## Clinical Impact

- Reduces diagnostic burden on mental health professionals
- Objective screening tool for primary care
- Early detection in at-risk populations
- Scalable to telehealth platforms

### To-dos

- [ ] Create project structure, requirements.txt with all dependencies, and initial README
- [ ] Download and organize Voiceome, DAIC-WOZ, and Vocal Mind datasets in /data directory
- [ ] Build audio preprocessing pipeline with normalization, VAD, and 60-second segmentation
- [ ] Implement eGeMAPS extraction with openSMILE and additional prosodic/voice quality features
- [ ] Train ensemble baseline models (RF, XGBoost, SVM) for each disorder with cross-validation
- [ ] Implement multi-task HuBERT with fine-tuned classification heads for 4 disorders
- [ ] Create ensemble combining baseline + HuBERT with confidence scoring and calibration
- [ ] Build interpretability module with SHAP values and attention visualization
- [ ] Develop Flask API with /analyze endpoint for audio upload and inference
- [ ] Create web interface with audio recorder, visualization, and results dashboard
- [ ] Generate clinical validation notebook with performance metrics and ROC curves
- [ ] Complete comprehensive README with usage, citations, and clinical guidelines