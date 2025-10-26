"""
Voice quality feature extraction for mental health screening.
Focuses on acoustic quality metrics like jitter, shimmer, and HNR.
"""

import numpy as np
import librosa
from scipy import signal
from scipy.stats import entropy
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class VoiceQualityExtractor:
    """Advanced voice quality feature extraction."""
    
    def __init__(self, sr: int = 16000):
        self.sr = sr
    
    def extract_jitter_shimmer(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract jitter and shimmer measures.
        
        Args:
            audio: Audio array
            
        Returns:
            Dictionary with jitter and shimmer features
        """
        features = {}
        
        try:
            # Extract F0
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=self.sr
            )
            
            f0_voiced = f0[voiced_flag]
            
            if len(f0_voiced) > 2:
                # Jitter calculations
                periods = 1.0 / f0_voiced
                
                # Absolute jitter
                period_diffs = np.abs(np.diff(periods))
                features['jitter_abs'] = np.mean(period_diffs)
                
                # Relative jitter
                features['jitter_rel'] = features['jitter_abs'] / np.mean(periods) if np.mean(periods) > 0 else 0
                
                # RAP (Relative Average Perturbation)
                if len(periods) >= 3:
                    rap_periods = []
                    for i in range(1, len(periods) - 1):
                        rap = abs(periods[i] - (periods[i-1] + periods[i+1]) / 2)
                        rap_periods.append(rap)
                    features['jitter_rap'] = np.mean(rap_periods) / np.mean(periods) if np.mean(periods) > 0 else 0
                else:
                    features['jitter_rap'] = 0
                
                # PPQ5 (5-point Period Perturbation Quotient)
                if len(periods) >= 5:
                    ppq5_periods = []
                    for i in range(2, len(periods) - 2):
                        ppq5 = abs(periods[i] - np.mean([periods[i-2], periods[i-1], periods[i+1], periods[i+2]]))
                        ppq5_periods.append(ppq5)
                    features['jitter_ppq5'] = np.mean(ppq5_periods) / np.mean(periods) if np.mean(periods) > 0 else 0
                else:
                    features['jitter_ppq5'] = 0
                
                # Shimmer calculations
                # Extract amplitude envelope
                amplitude_envelope = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
                
                # Interpolate to match F0 length
                if len(amplitude_envelope) != len(f0_voiced):
                    amplitude_envelope = np.interp(
                        np.linspace(0, 1, len(f0_voiced)),
                        np.linspace(0, 1, len(amplitude_envelope)),
                        amplitude_envelope
                    )
                
                # Absolute shimmer
                amp_diffs = np.abs(np.diff(amplitude_envelope))
                features['shimmer_abs'] = np.mean(amp_diffs)
                
                # Relative shimmer
                features['shimmer_rel'] = features['shimmer_abs'] / np.mean(amplitude_envelope) if np.mean(amplitude_envelope) > 0 else 0
                
                # APQ3 (3-point Amplitude Perturbation Quotient)
                if len(amplitude_envelope) >= 3:
                    apq3_amps = []
                    for i in range(1, len(amplitude_envelope) - 1):
                        apq3 = abs(amplitude_envelope[i] - (amplitude_envelope[i-1] + amplitude_envelope[i+1]) / 2)
                        apq3_amps.append(apq3)
                    features['shimmer_apq3'] = np.mean(apq3_amps) / np.mean(amplitude_envelope) if np.mean(amplitude_envelope) > 0 else 0
                else:
                    features['shimmer_apq3'] = 0
                
                # APQ5 (5-point Amplitude Perturbation Quotient)
                if len(amplitude_envelope) >= 5:
                    apq5_amps = []
                    for i in range(2, len(amplitude_envelope) - 2):
                        apq5 = abs(amplitude_envelope[i] - np.mean([amplitude_envelope[i-2], amplitude_envelope[i-1], 
                                                                   amplitude_envelope[i+1], amplitude_envelope[i+2]]))
                        apq5_amps.append(apq5)
                    features['shimmer_apq5'] = np.mean(apq5_amps) / np.mean(amplitude_envelope) if np.mean(amplitude_envelope) > 0 else 0
                else:
                    features['shimmer_apq5'] = 0
                
            else:
                # Default values for insufficient voiced segments
                for key in ['jitter_abs', 'jitter_rel', 'jitter_rap', 'jitter_ppq5',
                           'shimmer_abs', 'shimmer_rel', 'shimmer_apq3', 'shimmer_apq5']:
                    features[key] = 0.0
            
        except Exception as e:
            logger.error(f"Error extracting jitter/shimmer: {e}")
            for key in ['jitter_abs', 'jitter_rel', 'jitter_rap', 'jitter_ppq5',
                       'shimmer_abs', 'shimmer_rel', 'shimmer_apq3', 'shimmer_apq5']:
                features[key] = 0.0
        
        return features
    
    def extract_harmonic_noise_ratio(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract harmonic-to-noise ratio and related features.
        
        Args:
            audio: Audio array
            
        Returns:
            Dictionary with HNR and spectral features
        """
        features = {}
        
        try:
            # Extract F0
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=self.sr
            )
            
            f0_voiced = f0[voiced_flag]
            
            if len(f0_voiced) > 0:
                # Simplified HNR calculation using cepstral analysis
                # Compute cepstrum
                cepstrum = np.fft.ifft(np.log(np.abs(np.fft.fft(audio)) + 1e-8))
                cepstrum = np.real(cepstrum)
                
                # Find peak in cepstrum (corresponds to fundamental frequency)
                quefrency_range = int(self.sr / 50)  # Up to 50 Hz
                if quefrency_range < len(cepstrum):
                    cepstrum_peak = np.max(cepstrum[:quefrency_range])
                    cepstrum_noise = np.mean(cepstrum[quefrency_range:])
                    
                    if cepstrum_noise > 0:
                        features['hnr_cepstral'] = 20 * np.log10(cepstrum_peak / cepstrum_noise)
                    else:
                        features['hnr_cepstral'] = 0
                else:
                    features['hnr_cepstral'] = 0
                
                # Spectral features for HNR approximation
                # Spectral centroid
                spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sr)[0]
                features['spectral_centroid_mean'] = np.mean(spectral_centroids)
                features['spectral_centroid_std'] = np.std(spectral_centroids)
                
                # Spectral rolloff
                spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sr)[0]
                features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
                features['spectral_rolloff_std'] = np.std(spectral_rolloff)
                
                # Spectral bandwidth
                spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sr)[0]
                features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
                features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
                
                # Spectral contrast
                spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sr)
                features['spectral_contrast_mean'] = np.mean(spectral_contrast)
                features['spectral_contrast_std'] = np.std(spectral_contrast)
                
                # Spectral flatness (measure of noisiness)
                spectral_flatness = librosa.feature.spectral_flatness(y=audio)[0]
                features['spectral_flatness_mean'] = np.mean(spectral_flatness)
                features['spectral_flatness_std'] = np.std(spectral_flatness)
                
                # Zero crossing rate (indicator of noisiness)
                zcr = librosa.feature.zero_crossing_rate(audio)[0]
                features['zcr_mean'] = np.mean(zcr)
                features['zcr_std'] = np.std(zcr)
                
                # HNR approximation using spectral features
                # Higher spectral centroid and lower spectral flatness indicate more harmonic content
                if features['spectral_flatness_mean'] > 0:
                    features['hnr_approx'] = features['spectral_centroid_mean'] / (features['spectral_flatness_mean'] + 1e-8)
                else:
                    features['hnr_approx'] = 0
                
            else:
                # Default values for unvoiced audio
                for key in ['hnr_cepstral', 'spectral_centroid_mean', 'spectral_centroid_std',
                           'spectral_rolloff_mean', 'spectral_rolloff_std', 'spectral_bandwidth_mean',
                           'spectral_bandwidth_std', 'spectral_contrast_mean', 'spectral_contrast_std',
                           'spectral_flatness_mean', 'spectral_flatness_std', 'zcr_mean', 'zcr_std', 'hnr_approx']:
                    features[key] = 0.0
            
        except Exception as e:
            logger.error(f"Error extracting HNR features: {e}")
            for key in ['hnr_cepstral', 'spectral_centroid_mean', 'spectral_centroid_std',
                       'spectral_rolloff_mean', 'spectral_rolloff_std', 'spectral_bandwidth_mean',
                       'spectral_bandwidth_std', 'spectral_contrast_mean', 'spectral_contrast_std',
                       'spectral_flatness_mean', 'spectral_flatness_std', 'zcr_mean', 'zcr_std', 'hnr_approx']:
                features[key] = 0.0
        
        return features
    
    def extract_formant_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract formant-related features.
        
        Args:
            audio: Audio array
            
        Returns:
            Dictionary with formant features
        """
        features = {}
        
        try:
            # Extract F0 for voiced segments
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=self.sr
            )
            
            f0_voiced = f0[voiced_flag]
            
            if len(f0_voiced) > 0:
                # Formant estimation using LPC
                # Apply pre-emphasis
                pre_emphasized = librosa.effects.preemphasis(audio)
                
                # Extract formants using LPC
                # Use a window size that captures formant structure
                window_size = int(0.025 * self.sr)  # 25ms window
                hop_size = int(0.010 * self.sr)     # 10ms hop
                
                formant_frequencies = []
                formant_bandwidths = []
                
                for i in range(0, len(pre_emphasized) - window_size, hop_size):
                    window = pre_emphasized[i:i + window_size]
                    
                    # Apply window function
                    window = window * signal.windows.hann(len(window))
                    
                    # LPC analysis
                    lpc_order = min(12, len(window) - 1)
                    if lpc_order > 0:
                        try:
                            lpc_coeffs = librosa.lpc(window, order=lpc_order)
                            
                            # Find roots of LPC polynomial
                            roots = np.roots(lpc_coeffs)
                            
                            # Extract formant frequencies and bandwidths
                            formants = []
                            bandwidths = []
                            
                            for root in roots:
                                if np.iscomplex(root) and np.imag(root) > 0:
                                    # Convert to frequency and bandwidth
                                    freq = np.angle(root) * self.sr / (2 * np.pi)
                                    bandwidth = -np.log(np.abs(root)) * self.sr / np.pi
                                    
                                    # Filter reasonable formant range (80-4000 Hz)
                                    if 80 < freq < 4000 and bandwidth < 1000:
                                        formants.append(freq)
                                        bandwidths.append(bandwidth)
                            
                            # Sort formants by frequency
                            if formants:
                                sorted_indices = np.argsort(formants)
                                formants = [formants[i] for i in sorted_indices]
                                bandwidths = [bandwidths[i] for i in sorted_indices]
                                
                                # Take first 4 formants
                                for j in range(min(4, len(formants))):
                                    formant_frequencies.append(formants[j])
                                    formant_bandwidths.append(bandwidths[j])
                        
                        except:
                            continue
                
                # Calculate formant statistics
                if formant_frequencies:
                    # F1 (first formant)
                    f1_freqs = [f for f in formant_frequencies if 200 < f < 1000]
                    if f1_freqs:
                        features['f1_mean'] = np.mean(f1_freqs)
                        features['f1_std'] = np.std(f1_freqs)
                    else:
                        features['f1_mean'] = 0
                        features['f1_std'] = 0
                    
                    # F2 (second formant)
                    f2_freqs = [f for f in formant_frequencies if 800 < f < 2500]
                    if f2_freqs:
                        features['f2_mean'] = np.mean(f2_freqs)
                        features['f2_std'] = np.std(f2_freqs)
                    else:
                        features['f2_mean'] = 0
                        features['f2_std'] = 0
                    
                    # F3 (third formant)
                    f3_freqs = [f for f in formant_frequencies if 1500 < f < 3500]
                    if f3_freqs:
                        features['f3_mean'] = np.mean(f3_freqs)
                        features['f3_std'] = np.std(f3_freqs)
                    else:
                        features['f3_mean'] = 0
                        features['f3_std'] = 0
                    
                    # Formant bandwidths
                    if formant_bandwidths:
                        features['formant_bandwidth_mean'] = np.mean(formant_bandwidths)
                        features['formant_bandwidth_std'] = np.std(formant_bandwidths)
                    else:
                        features['formant_bandwidth_mean'] = 0
                        features['formant_bandwidth_std'] = 0
                    
                    # F2-F1 distance (vowel space)
                    if f1_freqs and f2_freqs:
                        features['f2_f1_distance'] = np.mean(f2_freqs) - np.mean(f1_freqs)
                    else:
                        features['f2_f1_distance'] = 0
                    
                    # Formant dispersion
                    all_formants = sorted([f for f in formant_frequencies if 200 < f < 4000])
                    if len(all_formants) >= 2:
                        formant_diffs = np.diff(all_formants)
                        features['formant_dispersion'] = np.mean(formant_diffs)
                    else:
                        features['formant_dispersion'] = 0
                
                else:
                    # Default values when no formants detected
                    for key in ['f1_mean', 'f1_std', 'f2_mean', 'f2_std', 'f3_mean', 'f3_std',
                               'formant_bandwidth_mean', 'formant_bandwidth_std', 'f2_f1_distance', 'formant_dispersion']:
                        features[key] = 0.0
            
            else:
                # Default values for unvoiced audio
                for key in ['f1_mean', 'f1_std', 'f2_mean', 'f2_std', 'f3_mean', 'f3_std',
                           'formant_bandwidth_mean', 'formant_bandwidth_std', 'f2_f1_distance', 'formant_dispersion']:
                    features[key] = 0.0
            
        except Exception as e:
            logger.error(f"Error extracting formant features: {e}")
            for key in ['f1_mean', 'f1_std', 'f2_mean', 'f2_std', 'f3_mean', 'f3_std',
                       'formant_bandwidth_mean', 'formant_bandwidth_std', 'f2_f1_distance', 'formant_dispersion']:
                features[key] = 0.0
        
        return features
    
    def extract_all_voice_quality_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract all voice quality features.
        
        Args:
            audio: Audio array
            
        Returns:
            Dictionary with all voice quality features
        """
        all_features = {}
        
        # Extract different types of voice quality features
        jitter_shimmer_features = self.extract_jitter_shimmer(audio)
        all_features.update(jitter_shimmer_features)
        
        hnr_features = self.extract_harmonic_noise_ratio(audio)
        all_features.update(hnr_features)
        
        formant_features = self.extract_formant_features(audio)
        all_features.update(formant_features)
        
        return all_features
