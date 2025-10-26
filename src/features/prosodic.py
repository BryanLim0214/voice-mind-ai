"""
Advanced prosodic feature extraction for voice analysis.
Focuses on pitch, timing, and rhythm patterns.
"""

import numpy as np
import librosa
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class ProsodicFeatureExtractor:
    """Advanced prosodic feature extraction."""
    
    def __init__(self, sr: int = 16000):
        self.sr = sr
    
    def extract_pitch_contour_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract detailed pitch contour features.
        
        Args:
            audio: Audio array
            
        Returns:
            Dictionary with pitch contour features
        """
        features = {}
        
        try:
            # Extract F0 with high precision
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=self.sr,
                frame_length=2048,
                hop_length=512
            )
            
            f0_voiced = f0[voiced_flag]
            
            if len(f0_voiced) > 10:  # Need sufficient voiced frames
                # Pitch contour statistics
                features['pitch_mean'] = np.mean(f0_voiced)
                features['pitch_std'] = np.std(f0_voiced)
                features['pitch_median'] = np.median(f0_voiced)
                features['pitch_25th_percentile'] = np.percentile(f0_voiced, 25)
                features['pitch_75th_percentile'] = np.percentile(f0_voiced, 75)
                features['pitch_iqr'] = features['pitch_75th_percentile'] - features['pitch_25th_percentile']
                
                # Pitch range and dynamics
                features['pitch_range'] = np.max(f0_voiced) - np.min(f0_voiced)
                features['pitch_range_semitones'] = 12 * np.log2(np.max(f0_voiced) / np.min(f0_voiced))
                
                # Pitch variation measures
                features['pitch_cv'] = features['pitch_std'] / features['pitch_mean'] if features['pitch_mean'] > 0 else 0
                
                # Pitch slope analysis
                pitch_diff = np.diff(f0_voiced)
                features['pitch_slope_mean'] = np.mean(pitch_diff)
                features['pitch_slope_std'] = np.std(pitch_diff)
                
                # Rising vs falling patterns
                rising_frames = np.sum(pitch_diff > 0)
                falling_frames = np.sum(pitch_diff < 0)
                total_frames = len(pitch_diff)
                
                features['pitch_rising_ratio'] = rising_frames / total_frames if total_frames > 0 else 0
                features['pitch_falling_ratio'] = falling_frames / total_frames if total_frames > 0 else 0
                
                # Pitch stability (inverse of variation)
                features['pitch_stability'] = 1.0 / (1.0 + features['pitch_cv'])
                
                # Micro-prosodic features
                # Jitter (cycle-to-cycle pitch variation)
                periods = 1.0 / f0_voiced
                period_diffs = np.abs(np.diff(periods))
                features['jitter_abs'] = np.mean(period_diffs)
                features['jitter_rel'] = features['jitter_abs'] / np.mean(periods) if np.mean(periods) > 0 else 0
                
                # Pitch acceleration
                if len(pitch_diff) > 1:
                    pitch_accel = np.diff(pitch_diff)
                    features['pitch_acceleration_mean'] = np.mean(pitch_accel)
                    features['pitch_acceleration_std'] = np.std(pitch_accel)
                else:
                    features['pitch_acceleration_mean'] = 0
                    features['pitch_acceleration_std'] = 0
                
            else:
                # Default values for insufficient voiced segments
                default_keys = [
                    'pitch_mean', 'pitch_std', 'pitch_median', 'pitch_25th_percentile',
                    'pitch_75th_percentile', 'pitch_iqr', 'pitch_range', 'pitch_range_semitones',
                    'pitch_cv', 'pitch_slope_mean', 'pitch_slope_std', 'pitch_rising_ratio',
                    'pitch_falling_ratio', 'pitch_stability', 'jitter_abs', 'jitter_rel',
                    'pitch_acceleration_mean', 'pitch_acceleration_std'
                ]
                for key in default_keys:
                    features[key] = 0.0
            
        except Exception as e:
            logger.error(f"Error extracting pitch contour features: {e}")
            # Return default values
            for key in ['pitch_mean', 'pitch_std', 'pitch_median', 'pitch_25th_percentile',
                       'pitch_75th_percentile', 'pitch_iqr', 'pitch_range', 'pitch_range_semitones',
                       'pitch_cv', 'pitch_slope_mean', 'pitch_slope_std', 'pitch_rising_ratio',
                       'pitch_falling_ratio', 'pitch_stability', 'jitter_abs', 'jitter_rel',
                       'pitch_acceleration_mean', 'pitch_acceleration_std']:
                features[key] = 0.0
        
        return features
    
    def extract_timing_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract timing and rhythm features.
        
        Args:
            audio: Audio array
            
        Returns:
            Dictionary with timing features
        """
        features = {}
        
        try:
            # Voice activity detection
            rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
            threshold = np.percentile(rms, 20)
            voice_frames = rms > threshold
            
            # Convert to time
            hop_duration = 512 / self.sr
            voice_times = voice_frames * hop_duration
            
            # Speech rate measures
            total_duration = len(audio) / self.sr
            voice_duration = np.sum(voice_frames) * hop_duration
            
            features['speech_rate'] = voice_duration / total_duration if total_duration > 0 else 0
            features['voice_activity_ratio'] = voice_duration / total_duration if total_duration > 0 else 0
            features['silence_ratio'] = 1.0 - features['voice_activity_ratio']
            
            # Articulation rate (approximation using spectral features)
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sr)[0]
            # Higher spectral centroid variation might indicate faster articulation
            features['articulation_rate_approx'] = np.std(spectral_centroids) / np.mean(spectral_centroids) if np.mean(spectral_centroids) > 0 else 0
            
            # Pause analysis
            voice_segments = self._find_voice_segments(voice_frames)
            
            if len(voice_segments) > 1:
                # Calculate pause durations
                pause_durations = []
                for i in range(len(voice_segments) - 1):
                    pause_duration = (voice_segments[i+1][0] - voice_segments[i][1]) * hop_duration
                    if pause_duration > 0.1:  # Pauses longer than 100ms
                        pause_durations.append(pause_duration)
                
                if pause_durations:
                    features['pause_count'] = len(pause_durations)
                    features['pause_mean_duration'] = np.mean(pause_durations)
                    features['pause_std_duration'] = np.std(pause_durations)
                    features['pause_total_duration'] = np.sum(pause_durations)
                    features['pause_ratio'] = features['pause_total_duration'] / total_duration if total_duration > 0 else 0
                    
                    # Pause distribution
                    features['pause_median_duration'] = np.median(pause_durations)
                    features['pause_max_duration'] = np.max(pause_durations)
                    features['pause_min_duration'] = np.min(pause_durations)
                else:
                    features['pause_count'] = 0
                    features['pause_mean_duration'] = 0
                    features['pause_std_duration'] = 0
                    features['pause_total_duration'] = 0
                    features['pause_ratio'] = 0
                    features['pause_median_duration'] = 0
                    features['pause_max_duration'] = 0
                    features['pause_min_duration'] = 0
            else:
                # No pauses detected
                for key in ['pause_count', 'pause_mean_duration', 'pause_std_duration',
                           'pause_total_duration', 'pause_ratio', 'pause_median_duration',
                           'pause_max_duration', 'pause_min_duration']:
                    features[key] = 0.0
            
            # Rhythm features
            # Tempo estimation
            tempo, beats = librosa.beat.beat_track(y=audio, sr=self.sr)
            features['tempo'] = tempo
            
            # Rhythm regularity (coefficient of variation of inter-beat intervals)
            if len(beats) > 2:
                beat_intervals = np.diff(beats) / self.sr
                features['rhythm_regularity'] = 1.0 / (1.0 + np.std(beat_intervals) / np.mean(beat_intervals)) if np.mean(beat_intervals) > 0 else 0
            else:
                features['rhythm_regularity'] = 0
            
        except Exception as e:
            logger.error(f"Error extracting timing features: {e}")
            # Return default values
            for key in ['speech_rate', 'voice_activity_ratio', 'silence_ratio', 'articulation_rate_approx',
                       'pause_count', 'pause_mean_duration', 'pause_std_duration', 'pause_total_duration',
                       'pause_ratio', 'pause_median_duration', 'pause_max_duration', 'pause_min_duration',
                       'tempo', 'rhythm_regularity']:
                features[key] = 0.0
        
        return features
    
    def extract_rhythm_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract rhythm and stress pattern features.
        
        Args:
            audio: Audio array
            
        Returns:
            Dictionary with rhythm features
        """
        features = {}
        
        try:
            # Onset detection for rhythm analysis
            onsets = librosa.onset.onset_detect(y=audio, sr=self.sr, units='time')
            
            if len(onsets) > 2:
                # Inter-onset intervals
                ioi = np.diff(onsets)
                
                # Rhythm statistics
                features['onset_count'] = len(onsets)
                features['onset_rate'] = len(onsets) / (len(audio) / self.sr)
                features['ioi_mean'] = np.mean(ioi)
                features['ioi_std'] = np.std(ioi)
                features['ioi_cv'] = features['ioi_std'] / features['ioi_mean'] if features['ioi_mean'] > 0 else 0
                
                # Rhythm regularity
                features['rhythm_regularity_ioi'] = 1.0 / (1.0 + features['ioi_cv'])
                
                # Stress pattern analysis (using energy peaks)
                rms = librosa.feature.rms(y=audio)[0]
                energy_peaks, _ = librosa.util.peak_pick(rms, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.1, wait=10)
                
                if len(energy_peaks) > 1:
                    stress_intervals = np.diff(energy_peaks) * (len(audio) / len(rms) / self.sr)
                    features['stress_rate'] = len(energy_peaks) / (len(audio) / self.sr)
                    features['stress_interval_mean'] = np.mean(stress_intervals)
                    features['stress_interval_std'] = np.std(stress_intervals)
                else:
                    features['stress_rate'] = 0
                    features['stress_interval_mean'] = 0
                    features['stress_interval_std'] = 0
            else:
                # Insufficient onsets
                for key in ['onset_count', 'onset_rate', 'ioi_mean', 'ioi_std', 'ioi_cv',
                           'rhythm_regularity_ioi', 'stress_rate', 'stress_interval_mean', 'stress_interval_std']:
                    features[key] = 0.0
            
            # Spectral flux for rhythm analysis
            spectral_flux = librosa.onset.onset_strength(y=audio, sr=self.sr)
            features['spectral_flux_mean'] = np.mean(spectral_flux)
            features['spectral_flux_std'] = np.std(spectral_flux)
            features['spectral_flux_max'] = np.max(spectral_flux)
            
        except Exception as e:
            logger.error(f"Error extracting rhythm features: {e}")
            # Return default values
            for key in ['onset_count', 'onset_rate', 'ioi_mean', 'ioi_std', 'ioi_cv',
                       'rhythm_regularity_ioi', 'stress_rate', 'stress_interval_mean', 'stress_interval_std',
                       'spectral_flux_mean', 'spectral_flux_std', 'spectral_flux_max']:
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
    
    def extract_all_prosodic_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract all prosodic features.
        
        Args:
            audio: Audio array
            
        Returns:
            Dictionary with all prosodic features
        """
        all_features = {}
        
        # Extract different types of prosodic features
        pitch_features = self.extract_pitch_contour_features(audio)
        all_features.update(pitch_features)
        
        timing_features = self.extract_timing_features(audio)
        all_features.update(timing_features)
        
        rhythm_features = self.extract_rhythm_features(audio)
        all_features.update(rhythm_features)
        
        return all_features
