"""
Feature extraction for multi-disorder voice screening.
Implements eGeMAPS extraction and additional acoustic features.
"""

import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import opensmile
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class VoiceFeatureExtractor:
    """Extracts comprehensive voice features for mental health screening."""
    
    def __init__(self, sr: int = 16000):
        """
        Initialize feature extractor.
        
        Args:
            sr: Sampling rate for audio processing
        """
        self.sr = sr
        
        # Initialize openSMILE for eGeMAPS
        try:
            self.smile = opensmile.Smile(
                feature_set=opensmile.FeatureSet.eGeMAPSv02,
                feature_level=opensmile.FeatureLevel.Functionals
            )
            logger.info("openSMILE eGeMAPS initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize openSMILE: {e}")
            self.smile = None
    
    def extract_egemaps_features(self, audio_path: str) -> Optional[pd.Series]:
        """
        Extract eGeMAPS features using openSMILE.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Series with eGeMAPS features or None if failed
        """
        if self.smile is None:
            logger.warning("openSMILE not available, skipping eGeMAPS features")
            return None
        
        try:
            features = self.smile.process_file(audio_path)
            # Flatten the features to a single row
            if len(features) > 0:
                return features.iloc[0]
            else:
                logger.warning(f"No eGeMAPS features extracted from {audio_path}")
                return None
        except Exception as e:
            logger.error(f"Error extracting eGeMAPS features from {audio_path}: {e}")
            return None
    
    def extract_prosodic_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract prosodic features (pitch, timing, rhythm).
        
        Args:
            audio: Audio array
            
        Returns:
            Dictionary with prosodic features
        """
        features = {}
        
        try:
            # Fundamental frequency (F0) extraction
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio, 
                fmin=librosa.note_to_hz('C2'), 
                fmax=librosa.note_to_hz('C7'),
                sr=self.sr
            )
            
            # Remove unvoiced segments
            f0_voiced = f0[voiced_flag]
            
            if len(f0_voiced) > 0:
                # Pitch statistics
                features['f0_mean'] = np.mean(f0_voiced)
                features['f0_std'] = np.std(f0_voiced)
                features['f0_median'] = np.median(f0_voiced)
                features['f0_min'] = np.min(f0_voiced)
                features['f0_max'] = np.max(f0_voiced)
                features['f0_range'] = features['f0_max'] - features['f0_min']
                
                # Pitch variation measures
                features['f0_cv'] = features['f0_std'] / features['f0_mean'] if features['f0_mean'] > 0 else 0
                features['f0_jitter'] = np.mean(np.abs(np.diff(f0_voiced))) / features['f0_mean'] if features['f0_mean'] > 0 else 0
                
                # Pitch contour features
                f0_diff = np.diff(f0_voiced)
                features['f0_rising_slopes'] = np.sum(f0_diff > 0) / len(f0_diff)
                features['f0_falling_slopes'] = np.sum(f0_diff < 0) / len(f0_diff)
                
            else:
                # Default values for unvoiced audio
                for key in ['f0_mean', 'f0_std', 'f0_median', 'f0_min', 'f0_max', 'f0_range', 
                           'f0_cv', 'f0_jitter', 'f0_rising_slopes', 'f0_falling_slopes']:
                    features[key] = 0.0
            
            # Speech rate and timing
            # Voice activity detection for speech rate
            rms = librosa.feature.rms(y=audio)[0]
            threshold = np.percentile(rms, 20)
            voice_frames = rms > threshold
            
            # Calculate speech rate (syllables per second approximation)
            frame_duration = 0.025  # 25ms frames
            total_duration = len(audio) / self.sr
            voice_duration = np.sum(voice_frames) * frame_duration
            
            features['speech_rate'] = voice_duration / total_duration if total_duration > 0 else 0
            features['voice_activity_ratio'] = voice_duration / total_duration if total_duration > 0 else 0
            
            # Pause analysis
            voice_segments = self._find_voice_segments(voice_frames)
            if len(voice_segments) > 1:
                pause_durations = []
                for i in range(len(voice_segments) - 1):
                    pause_duration = (voice_segments[i+1][0] - voice_segments[i][1]) * frame_duration
                    if pause_duration > 0.1:  # Pauses longer than 100ms
                        pause_durations.append(pause_duration)
                
                if pause_durations:
                    features['pause_count'] = len(pause_durations)
                    features['pause_mean_duration'] = np.mean(pause_durations)
                    features['pause_std_duration'] = np.std(pause_durations)
                else:
                    features['pause_count'] = 0
                    features['pause_mean_duration'] = 0
                    features['pause_std_duration'] = 0
            else:
                features['pause_count'] = 0
                features['pause_mean_duration'] = 0
                features['pause_std_duration'] = 0
            
        except Exception as e:
            logger.error(f"Error extracting prosodic features: {e}")
            # Return default values
            for key in ['f0_mean', 'f0_std', 'f0_median', 'f0_min', 'f0_max', 'f0_range', 
                       'f0_cv', 'f0_jitter', 'f0_rising_slopes', 'f0_falling_slopes',
                       'speech_rate', 'voice_activity_ratio', 'pause_count', 
                       'pause_mean_duration', 'pause_std_duration']:
                features[key] = 0.0
        
        return features
    
    def extract_voice_quality_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract voice quality features (jitter, shimmer, HNR).
        
        Args:
            audio: Audio array
            
        Returns:
            Dictionary with voice quality features
        """
        features = {}
        
        try:
            # Extract F0 for voice quality analysis
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio, 
                fmin=librosa.note_to_hz('C2'), 
                fmax=librosa.note_to_hz('C7'),
                sr=self.sr
            )
            
            f0_voiced = f0[voiced_flag]
            
            if len(f0_voiced) > 2:
                # Jitter (pitch period perturbation)
                periods = 1.0 / f0_voiced
                period_diffs = np.abs(np.diff(periods))
                features['jitter_abs'] = np.mean(period_diffs)
                features['jitter_rel'] = features['jitter_abs'] / np.mean(periods) if np.mean(periods) > 0 else 0
                
                # Shimmer (amplitude perturbation)
                # Extract amplitude envelope
                amplitude_envelope = librosa.feature.rms(y=audio)[0]
                # Interpolate to match F0 length
                if len(amplitude_envelope) != len(f0_voiced):
                    amplitude_envelope = np.interp(
                        np.linspace(0, 1, len(f0_voiced)),
                        np.linspace(0, 1, len(amplitude_envelope)),
                        amplitude_envelope
                    )
                
                amp_diffs = np.abs(np.diff(amplitude_envelope))
                features['shimmer_abs'] = np.mean(amp_diffs)
                features['shimmer_rel'] = features['shimmer_abs'] / np.mean(amplitude_envelope) if np.mean(amplitude_envelope) > 0 else 0
                
                # Harmonic-to-Noise Ratio (HNR)
                # Simplified HNR calculation using spectral centroid
                spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sr)[0]
                spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sr)[0]
                
                # HNR approximation using spectral features
                features['hnr_approx'] = np.mean(spectral_centroids) / (np.mean(spectral_rolloff) + 1e-8)
                
            else:
                # Default values for insufficient voiced segments
                for key in ['jitter_abs', 'jitter_rel', 'shimmer_abs', 'shimmer_rel', 'hnr_approx']:
                    features[key] = 0.0
            
            # Spectral features
            # Spectral centroid (brightness)
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sr)[0]
            features['spectral_centroid_mean'] = np.mean(spectral_centroids)
            features['spectral_centroid_std'] = np.std(spectral_centroids)
            
            # Spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sr)[0]
            features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
            features['spectral_rolloff_std'] = np.std(spectral_rolloff)
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            features['zcr_mean'] = np.mean(zcr)
            features['zcr_std'] = np.std(zcr)
            
            # MFCC features (first 4 coefficients)
            mfccs = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=13)
            for i in range(4):
                features[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])
                features[f'mfcc_{i+1}_std'] = np.std(mfccs[i])
            
        except Exception as e:
            logger.error(f"Error extracting voice quality features: {e}")
            # Return default values
            for key in ['jitter_abs', 'jitter_rel', 'shimmer_abs', 'shimmer_rel', 'hnr_approx',
                       'spectral_centroid_mean', 'spectral_centroid_std', 'spectral_rolloff_mean',
                       'spectral_rolloff_std', 'zcr_mean', 'zcr_std']:
                features[key] = 0.0
            for i in range(4):
                features[f'mfcc_{i+1}_mean'] = 0.0
                features[f'mfcc_{i+1}_std'] = 0.0
        
        return features
    
    def extract_energy_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract energy-related features.
        
        Args:
            audio: Audio array
            
        Returns:
            Dictionary with energy features
        """
        features = {}
        
        try:
            # RMS energy
            rms = librosa.feature.rms(y=audio)[0]
            features['rms_mean'] = np.mean(rms)
            features['rms_std'] = np.std(rms)
            features['rms_max'] = np.max(rms)
            features['rms_min'] = np.min(rms)
            features['rms_range'] = features['rms_max'] - features['rms_min']
            
            # Energy dynamics
            features['energy_entropy'] = stats.entropy(rms + 1e-8)
            
            # Energy variation
            rms_diff = np.diff(rms)
            features['energy_variation'] = np.std(rms_diff)
            
            # Loudness (approximated by RMS)
            features['loudness_mean'] = features['rms_mean']
            features['loudness_std'] = features['rms_std']
            
        except Exception as e:
            logger.error(f"Error extracting energy features: {e}")
            for key in ['rms_mean', 'rms_std', 'rms_max', 'rms_min', 'rms_range',
                       'energy_entropy', 'energy_variation', 'loudness_mean', 'loudness_std']:
                features[key] = 0.0
        
        return features
    
    def _find_voice_segments(self, voice_frames: np.ndarray) -> List[Tuple[int, int]]:
        """
        Find continuous voice segments from voice activity frames.
        
        Args:
            voice_frames: Boolean array indicating voice activity
            
        Returns:
            List of (start, end) frame indices for voice segments
        """
        segments = []
        in_segment = False
        start = 0
        
        for i, is_voice in enumerate(voice_frames):
            if is_voice and not in_segment:
                start = i
                in_segment = True
            elif not is_voice and in_segment:
                segments.append((start, i))
                in_segment = False
        
        # Handle case where segment continues to end
        if in_segment:
            segments.append((start, len(voice_frames)))
        
        return segments
    
    def extract_all_features(self, audio_path: str) -> Dict[str, float]:
        """
        Extract all features from an audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with all extracted features
        """
        all_features = {}
        
        try:
            # Load audio
            audio, sr = sf.read(audio_path)
            if sr != self.sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sr)
            
            # Extract eGeMAPS features
            egemaps_features = self.extract_egemaps_features(audio_path)
            if egemaps_features is not None:
                all_features.update(egemaps_features.to_dict())
            
            # Extract custom features
            prosodic_features = self.extract_prosodic_features(audio)
            all_features.update(prosodic_features)
            
            voice_quality_features = self.extract_voice_quality_features(audio)
            all_features.update(voice_quality_features)
            
            energy_features = self.extract_energy_features(audio)
            all_features.update(energy_features)
            
        except Exception as e:
            logger.error(f"Error extracting features from {audio_path}: {e}")
            return {}
        
        return all_features
    
    def extract_dataset_features(self, 
                                metadata_df: pd.DataFrame, 
                                output_path: str) -> pd.DataFrame:
        """
        Extract features for entire dataset.
        
        Args:
            metadata_df: DataFrame with participant metadata
            output_path: Path to save extracted features
            
        Returns:
            DataFrame with extracted features
        """
        feature_rows = []
        
        for idx, row in metadata_df.iterrows():
            participant_id = row['participant_id']
            dataset = row['dataset']
            
            # Construct audio file path
            audio_path = f"data/processed/{dataset}/{participant_id}_processed.wav"
            
            if os.path.exists(audio_path):
                logger.info(f"Extracting features for {participant_id}")
                features = self.extract_all_features(audio_path)
                
                if features:
                    # Add participant info
                    features['participant_id'] = participant_id
                    features['dataset'] = dataset
                    features['age'] = row.get('age', 0)
                    features['gender'] = row.get('gender', 'U')
                    
                    # Add labels
                    features['depression_label'] = row.get('depression_severity', 'none')
                    features['anxiety_label'] = row.get('anxiety_severity', 'none')
                    features['ptsd_label'] = row.get('ptsd_present', 'none')
                    features['cognitive_label'] = row.get('cognitive_status', 'normal')
                    
                    feature_rows.append(features)
                else:
                    logger.warning(f"No features extracted for {participant_id}")
            else:
                logger.warning(f"Audio file not found: {audio_path}")
        
        # Create features DataFrame
        if feature_rows:
            features_df = pd.DataFrame(feature_rows)
            
            # Save features
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            features_df.to_csv(output_path, index=False)
            
            logger.info(f"Extracted features for {len(features_df)} participants")
            logger.info(f"Features saved to {output_path}")
            
            return features_df
        else:
            logger.error("No features extracted")
            return pd.DataFrame()

def main():
    """Main feature extraction function."""
    # Load metadata
    metadata_path = "data/processed/unified_metadata.csv"
    if not os.path.exists(metadata_path):
        logger.error(f"Metadata file not found: {metadata_path}")
        return
    
    metadata_df = pd.read_csv(metadata_path)
    
    # Initialize feature extractor
    extractor = VoiceFeatureExtractor()
    
    # Extract features
    output_path = "data/features/extracted_features.csv"
    features_df = extractor.extract_dataset_features(metadata_df, output_path)
    
    if not features_df.empty:
        logger.info(f"Feature extraction completed. Shape: {features_df.shape}")
        logger.info(f"Features: {list(features_df.columns)}")

if __name__ == "__main__":
    main()
