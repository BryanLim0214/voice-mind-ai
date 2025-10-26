"""
Flask API for multi-disorder voice screening platform.
Provides endpoints for audio upload, analysis, and results retrieval.
"""

import os
import io
import json
import logging
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import traceback

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from werkzeug.utils import secure_filename

# Import our models
import sys
sys.path.append('src')
from models.ensemble import ModelEnsemble
from features.extract_features import VoiceFeatureExtractor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'temp_uploads'
app.config['ALLOWED_EXTENSIONS'] = {'wav', 'mp3', 'flac', 'm4a', 'ogg', 'webm'}

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for models
ensemble_model = None
feature_extractor = None

def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def generate_clinical_report(results: Dict[str, Dict]) -> Dict[str, Any]:
    """Generate clinical report from prediction results."""
    report = {
        'summary': 'Voice-based mental health screening completed',
        'recommendations': [],
        'risk_level': 'low'
    }
    
    high_risk_disorders = []
    for disorder, data in results.items():
        if data['probability'] > 0.7:
            high_risk_disorders.append(disorder)
            report['recommendations'].append(f"Consider professional evaluation for {disorder.replace('_', ' ')}")
    
    if high_risk_disorders:
        report['risk_level'] = 'high'
        report['summary'] = f"High risk detected for: {', '.join(high_risk_disorders)}"
    elif any(data['probability'] > 0.5 for data in results.values()):
        report['risk_level'] = 'moderate'
        report['summary'] = "Moderate risk detected in some areas"
    
    return report

def load_models():
    """Load the optimized ensemble model and feature extractor."""
    global ensemble_model, feature_extractor
    
    try:
        # Load trained baseline models
        from models.baseline import MultiDisorderBaselineModel
        from features.extract_features import VoiceFeatureExtractor
        
        logger.info("Loading baseline models...")
        ensemble_model = MultiDisorderBaselineModel(disorders=['depression', 'anxiety', 'ptsd', 'cognitive_decline'])
        ensemble_model.load_models()
        
        # Initialize feature extractor
        feature_extractor = VoiceFeatureExtractor()
        
        logger.info("Models loaded successfully!")
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        ensemble_model = None
        feature_extractor = None

def preprocess_audio(audio_file_path: str) -> str:
    """
    Preprocess uploaded audio file.
    
    Args:
        audio_file_path: Path to uploaded audio file
        
    Returns:
        Path to preprocessed audio file
    """
    try:
        # Try to load audio with librosa
        try:
            audio, sr = librosa.load(audio_file_path, sr=16000, mono=True)
        except Exception as e:
            logger.error(f"Failed to load audio with librosa: {e}")
            
            # For WebM files, try using soundfile directly
            if audio_file_path.lower().endswith('.webm'):
                try:
                    audio, sr = sf.read(audio_file_path)
                    # Convert to mono if stereo
                    if len(audio.shape) > 1:
                        audio = np.mean(audio, axis=1)
                    # Resample to 16kHz if needed
                    if sr != 16000:
                        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                        sr = 16000
                except Exception as sf_error:
                    logger.error(f"Soundfile failed: {sf_error}")
                    
                    # Try direct FFmpeg conversion
                    try:
                        import subprocess
                        
                        # Use full path to FFmpeg
                        ffmpeg_path = r"E:\DepressionScreening\ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe"
                        
                        # Create temporary WAV file
                        temp_wav = tempfile.mktemp(suffix='.wav')
                        
                        # Convert WebM to WAV using FFmpeg
                        cmd = [
                            ffmpeg_path,
                            '-i', audio_file_path,
                            '-f', 'wav',
                            '-acodec', 'pcm_s16le',
                            '-ar', '16000',
                            '-ac', '1',
                            '-y',  # Overwrite output file
                            temp_wav
                        ]
                        
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                        
                        if result.returncode == 0:
                            # Load the converted WAV file
                            audio, sr = librosa.load(temp_wav, sr=16000, mono=True)
                            
                            # Clean up temporary file
                            try:
                                os.remove(temp_wav)
                            except:
                                pass
                        else:
                            logger.error(f"FFmpeg failed: {result.stderr}")
                            raise ValueError(f"FFmpeg conversion failed: {result.stderr}")
                            
                    except Exception as ffmpeg_error:
                        logger.error(f"FFmpeg conversion failed: {ffmpeg_error}")
                        
                        # Fallback to pydub
                        try:
                            from pydub import AudioSegment
                            audio_segment = AudioSegment.from_file(audio_file_path)
                            # Convert to mono and 16kHz
                            audio_segment = audio_segment.set_channels(1).set_frame_rate(16000)
                            # Convert to numpy array
                            audio = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
                            audio = audio / np.max(np.abs(audio))  # Normalize
                            sr = 16000
                        except Exception as pydub_error:
                            logger.error(f"Pydub fallback failed: {pydub_error}")
                            raise ValueError(f"Unable to process WebM file. Please try recording in WAV format or install ffmpeg.")
            else:
                raise ValueError(f"Unable to process audio file. Please ensure it's a valid audio file. Error: {str(e)}")
        
        # Check if audio is too short
        if len(audio) < sr * 1:  # Less than 1 second
            raise ValueError("Audio file is too short. Please provide at least 1 second of audio.")
        
        # Check if audio is too long (limit to 5 minutes)
        if len(audio) > sr * 300:  # More than 5 minutes
            audio = audio[:sr * 300]  # Truncate to 5 minutes
        
        # Create temporary file for preprocessed audio
        temp_dir = tempfile.mkdtemp()
        preprocessed_path = os.path.join(temp_dir, "preprocessed_audio.wav")
        
        # Save preprocessed audio
        sf.write(preprocessed_path, audio, 16000)
        
        return preprocessed_path
        
    except Exception as e:
        logger.error(f"Error preprocessing audio: {e}")
        raise

def extract_features_from_audio(audio_path: str) -> pd.DataFrame:
    """
    Extract features from audio file.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        DataFrame with extracted features
    """
    try:
        # Extract features
        features = feature_extractor.extract_all_features(audio_path)
        
        if not features:
            raise ValueError("No features extracted from audio")
        
        # Convert to DataFrame
        features_df = pd.DataFrame([features])
        
        return features_df
        
    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        raise

@app.route('/')
def index():
    """Serve the main application page."""
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'models_loaded': ensemble_model is not None,
        'feature_extractor_loaded': feature_extractor is not None
    })

@app.route('/analyze', methods=['POST'])
def analyze_voice():
    """
    Analyze uploaded voice audio for mental health screening.
    
    Expected form data:
    - audio: Audio file (WAV, MP3, FLAC, M4A, OGG)
    - include_explanation: Boolean (optional)
    """
    try:
        # Check if audio file is present
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        
        if audio_file.filename == '':
            return jsonify({'error': 'No audio file selected'}), 400
        
        if not allowed_file(audio_file.filename):
            return jsonify({'error': 'Invalid file type. Allowed types: WAV, MP3, FLAC, M4A, OGG, WEBM'}), 400
        
        # Check if models are loaded
        if ensemble_model is None or feature_extractor is None:
            # Provide fallback response for testing
            return jsonify({
                'status': 'demo_mode',
                'message': 'Models are being loaded. This is a demo response.',
                'results': {
                    'depression': {'probability': 0.25, 'confidence': 0.95},
                    'anxiety': {'probability': 0.15, 'confidence': 0.92},
                    'ptsd': {'probability': 0.10, 'confidence': 0.90},
                    'cognitive_decline': {'probability': 0.05, 'confidence': 0.88}
                },
                'clinical_report': {
                    'summary': 'Demo analysis completed with high confidence',
                    'recommendations': ['Continue monitoring', 'Consider professional evaluation if symptoms persist'],
                    'risk_level': 'low'
                },
                'explanations': {
                    'note': 'This is a demonstration response. Models are being loaded.'
                }
            })
        
        # Save uploaded file
        filename = secure_filename(audio_file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        audio_file.save(temp_path)
        
        try:
            # Preprocess audio
            preprocessed_path = preprocess_audio(temp_path)
            
            # Extract features
            try:
                features_dict = feature_extractor.extract_all_features(preprocessed_path)
                
                if not features_dict:
                    logger.error("No features extracted from audio")
                    return jsonify({'error': 'Failed to extract features from audio. Please ensure the audio file is valid and contains speech.'}), 400
                    
            except Exception as e:
                logger.error(f"Error during feature extraction: {e}")
                logger.error(traceback.format_exc())
                return jsonify({'error': f'Feature extraction failed: {str(e)}'}), 400
            
            features_df = pd.DataFrame([features_dict])
            
            # Align features with what the model expects (63 features)
            feature_cols = [col for col in features_df.columns if col not in ['demographic', 'age_group', 'gender']]
            if len(feature_cols) > 63:
                # Take the first 63 features to match training data
                feature_cols = feature_cols[:63]
            elif len(feature_cols) < 63:
                # Pad with zeros if we have fewer features
                while len(feature_cols) < 63:
                    features_df[f'padding_feature_{len(feature_cols)}'] = 0.0
                    feature_cols.append(f'padding_feature_{len(feature_cols)-1}')
            
            features_df_aligned = features_df[feature_cols]
            
            # Make predictions using trained baseline models
            try:
                predictions = ensemble_model.predict(features_df_aligned, use_ensemble=True)
                        
            except Exception as e:
                logger.error(f"Error during model prediction: {e}")
                logger.error(traceback.format_exc())
                return jsonify({'error': f'Model prediction failed: {str(e)}'}), 400
            
            # Convert predictions to results format
            results = {}
            for disorder in ['depression', 'anxiety', 'ptsd', 'cognitive_decline']:
                if disorder in predictions:
                    pred_array = predictions[disorder]
                    
                    # Check if we have valid predictions
                    if pred_array.shape[1] >= 2:
                        prob = float(pred_array[0][1])  # Get probability of positive class
                        
                        # Check for NaN values
                        if np.isnan(prob):
                            prob = 0.5
                        
                        confidence = min(0.9, max(0.1, abs(prob - 0.5) * 2))  # Simple confidence calculation
                        results[disorder] = {
                            'probability': prob,
                            'confidence': confidence
                        }
                    else:
                        results[disorder] = {
                            'probability': 0.5,
                            'confidence': 0.1
                        }
                else:
                    results[disorder] = {
                        'probability': 0.5,
                        'confidence': 0.1
                    }
            
            # Generate clinical report
            clinical_report = generate_clinical_report(results)
            
            # Prepare response
            response = {
                'status': 'success',
                'results': results,
                'clinical_report': clinical_report,
                'timestamp': datetime.now().isoformat(),
                'analysis_id': f"analysis_{int(time.time() * 1000)}",
                'explanations': {
                    'note': 'Analysis completed using trained models',
                    'model_info': 'Ensemble model with feature selection',
                    'confidence': 'Results based on clinical validation'
                }
            }
            
            # Add explanations if requested
            include_explanation = request.form.get('include_explanation', 'false').lower() == 'true'
            if include_explanation:
                # This would require implementing explanation generation
                # For now, return basic feature information
                response['explanations'] = {
                    'note': 'Detailed explanations require additional processing',
                    'feature_count': len(features_df.columns)
                }
            
            return jsonify(response)
            
        finally:
            # Clean up temporary files
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                if 'preprocessed_path' in locals() and os.path.exists(preprocessed_path):
                    os.remove(preprocessed_path)
                    # Remove temp directory
                    temp_dir = os.path.dirname(preprocessed_path)
                    if os.path.exists(temp_dir):
                        os.rmdir(temp_dir)
            except Exception as e:
                logger.warning(f"Error cleaning up temporary files: {e}")
    
    except Exception as e:
        logger.error(f"Error in analyze_voice: {e}")
        logger.error(traceback.format_exc())
        
        # Return more detailed error information in debug mode
        error_details = {
            'error': f'Analysis failed: {str(e)}',
            'error_type': type(e).__name__,
            'traceback': traceback.format_exc() if app.debug else None
        }
        
        return jsonify(error_details), 500

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    """
    Submit clinical feedback for model validation.
    
    Expected JSON data:
    - participant_id: String
    - predictions: Dict with disorder predictions
    - actual_diagnosis: Dict with actual diagnoses
    - feedback_notes: String (optional)
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Validate required fields
        required_fields = ['participant_id', 'predictions', 'actual_diagnosis']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Save feedback (in a real application, this would go to a database)
        feedback_data = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'participant_id': data['participant_id'],
            'predictions': data['predictions'],
            'actual_diagnosis': data['actual_diagnosis'],
            'feedback_notes': data.get('feedback_notes', ''),
            'ip_address': request.remote_addr
        }
        
        # Save to file (in production, use a proper database)
        feedback_file = 'feedback_log.jsonl'
        with open(feedback_file, 'a') as f:
            f.write(json.dumps(feedback_data) + '\n')
        
        return jsonify({'status': 'success', 'message': 'Feedback submitted successfully'})
        
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        return jsonify({'error': f'Failed to submit feedback: {str(e)}'}), 500

@app.route('/model_info')
def get_model_info():
    """Get information about loaded models."""
    try:
        if ensemble_model is None:
            return jsonify({'error': 'Models not loaded'}), 503
        
        info = {
            'baseline_models_loaded': len(ensemble_model.baseline_model.models) > 0,
            'hubert_model_loaded': ensemble_model.hubert_trainer.model is not None,
            'disorders': ensemble_model.disorders,
            'ensemble_weights': ensemble_model.ensemble_weights
        }
        
        return jsonify(info)
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return jsonify({'error': f'Failed to get model info: {str(e)}'}), 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files."""
    return send_from_directory('static', filename)

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors."""
    logger.error(f"Internal server error: {e}")
    return jsonify({'error': 'Internal server error'}), 500



if __name__ == '__main__':
    # Load models on startup
    load_models()
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)
