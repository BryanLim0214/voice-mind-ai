"""
Audio preprocessing pipeline for multi-disorder voice screening.
Handles normalization, voice activity detection, and segmentation.
"""

import os
import librosa
import soundfile as sf
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioPreprocessor:
    """Handles audio preprocessing for voice screening datasets."""
    
    def __init__(self, 
                 target_sr: int = 16000,
                 target_duration: float = 60.0,
                 min_audio_duration: float = 5.0):
        """
        Initialize audio preprocessor.
        
        Args:
            target_sr: Target sampling rate (Hz)
            target_duration: Target audio duration (seconds)
            min_audio_duration: Minimum audio duration to process (seconds)
        """
        self.target_sr = target_sr
        self.target_duration = target_duration
        self.min_audio_duration = min_audio_duration
        
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file with librosa.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (audio_array, sampling_rate)
        """
        try:
            audio, sr = librosa.load(file_path, sr=self.target_sr, mono=True)
            return audio, sr
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return None, None
    
    def detect_voice_activity(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Simple voice activity detection using energy threshold.
        
        Args:
            audio: Audio array
            sr: Sampling rate
            
        Returns:
            Boolean array indicating voice activity
        """
        # Calculate frame energy
        frame_length = int(0.025 * sr)  # 25ms frames
        hop_length = int(0.010 * sr)    # 10ms hop
        
        # Compute RMS energy
        rms = librosa.feature.rms(y=audio, 
                                 frame_length=frame_length, 
                                 hop_length=hop_length)[0]
        
        # Threshold based on 20th percentile of energy
        threshold = np.percentile(rms, 20)
        voice_frames = rms > threshold
        
        # Convert back to sample-level
        voice_samples = np.repeat(voice_frames, hop_length)
        if len(voice_samples) < len(audio):
            voice_samples = np.pad(voice_samples, (0, len(audio) - len(voice_samples)))
        else:
            voice_samples = voice_samples[:len(audio)]
            
        return voice_samples
    
    def remove_silence(self, audio: np.ndarray, voice_activity: np.ndarray) -> np.ndarray:
        """
        Remove silence from audio based on voice activity detection.
        
        Args:
            audio: Audio array
            voice_activity: Boolean array indicating voice activity
            
        Returns:
            Audio with silence removed
        """
        # Find first and last voice activity
        voice_indices = np.where(voice_activity)[0]
        if len(voice_indices) == 0:
            return audio
            
        start_idx = max(0, voice_indices[0] - int(0.1 * self.target_sr))  # 100ms buffer
        end_idx = min(len(audio), voice_indices[-1] + int(0.1 * self.target_sr))
        
        return audio[start_idx:end_idx]
    
    def segment_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Segment audio to target duration.
        
        Args:
            audio: Audio array
            
        Returns:
            Segmented audio array
        """
        target_samples = int(self.target_duration * self.target_sr)
        
        if len(audio) >= target_samples:
            # Take middle segment for longer audio
            start_idx = (len(audio) - target_samples) // 2
            return audio[start_idx:start_idx + target_samples]
        else:
            # Pad shorter audio with silence
            padding = target_samples - len(audio)
            return np.pad(audio, (0, padding), mode='constant')
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to [-1, 1] range.
        
        Args:
            audio: Audio array
            
        Returns:
            Normalized audio array
        """
        if np.max(np.abs(audio)) > 0:
            return audio / np.max(np.abs(audio))
        return audio
    
    def preprocess_file(self, input_path: str, output_path: str) -> bool:
        """
        Preprocess a single audio file.
        
        Args:
            input_path: Input audio file path
            output_path: Output audio file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load audio
            audio, sr = self.load_audio(input_path)
            if audio is None:
                return False
            
            # Check minimum duration
            if len(audio) / sr < self.min_audio_duration:
                logger.warning(f"Audio too short: {input_path}")
                return False
            
            # Voice activity detection
            voice_activity = self.detect_voice_activity(audio, sr)
            
            # Remove silence
            audio = self.remove_silence(audio, voice_activity)
            
            # Check if audio is still long enough after silence removal
            if len(audio) / sr < self.min_audio_duration:
                logger.warning(f"Audio too short after silence removal: {input_path}")
                return False
            
            # Segment to target duration
            audio = self.segment_audio(audio)
            
            # Normalize
            audio = self.normalize_audio(audio)
            
            # Save processed audio
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            sf.write(output_path, audio, self.target_sr)
            
            return True
            
        except Exception as e:
            logger.error(f"Error preprocessing {input_path}: {e}")
            return False
    
    def preprocess_dataset(self, 
                          input_dir: str, 
                          output_dir: str,
                          file_extensions: List[str] = ['.wav', '.mp3', '.flac', '.m4a']) -> pd.DataFrame:
        """
        Preprocess entire dataset.
        
        Args:
            input_dir: Input directory containing audio files
            output_dir: Output directory for processed files
            file_extensions: Supported audio file extensions
            
        Returns:
            DataFrame with processing results
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # Find all audio files
        audio_files = []
        for ext in file_extensions:
            audio_files.extend(input_path.rglob(f"*{ext}"))
        
        logger.info(f"Found {len(audio_files)} audio files")
        
        # Process files
        results = []
        for audio_file in tqdm(audio_files, desc="Preprocessing audio"):
            # Create output path maintaining directory structure
            rel_path = audio_file.relative_to(input_path)
            output_file = output_path / rel_path.with_suffix('.wav')
            
            success = self.preprocess_file(str(audio_file), str(output_file))
            
            results.append({
                'input_path': str(audio_file),
                'output_path': str(output_file),
                'success': success,
                'original_duration': librosa.get_duration(filename=str(audio_file)) if success else None
            })
        
        # Create results DataFrame
        if results:
            results_df = pd.DataFrame(results)
            
            # Save processing log
            log_path = output_path / 'preprocessing_log.csv'
            results_df.to_csv(log_path, index=False)
            
            successful = results_df['success'].sum()
            logger.info(f"Successfully processed {successful}/{len(audio_files)} files")
            
            return results_df
        else:
            logger.warning("No audio files found to preprocess")
            return pd.DataFrame()

def create_label_mapping() -> Dict[str, Dict[str, int]]:
    """
    Create label mapping for multi-disorder classification.
    
    Returns:
        Dictionary mapping dataset names to disorder label mappings
    """
    label_mappings = {
        'voiceome': {
            'depression': {'none': 0, 'mild': 1, 'moderate': 2, 'severe': 3},
            'anxiety': {'none': 0, 'mild': 1, 'moderate': 2, 'severe': 3},
            'ptsd': {'none': 0, 'present': 1},
            'cognitive': {'normal': 0, 'mild_impairment': 1, 'moderate_impairment': 2}
        },
        'daic': {
            'depression': {'none': 0, 'mild': 1, 'moderate': 2, 'severe': 3},
            'anxiety': {'none': 0, 'mild': 1, 'moderate': 2, 'severe': 3},
            'ptsd': {'none': 0, 'present': 1}
        },
        'vocal_mind': {
            'depression': {'none': 0, 'mild': 1, 'moderate': 2, 'severe': 3}
        }
    }
    return label_mappings

def main():
    """Main preprocessing function."""
    # Initialize preprocessor
    preprocessor = AudioPreprocessor()
    
    # Define paths
    data_dir = Path("data")
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    
    # Process each dataset
    datasets = ['voiceome', 'daic', 'vocal_mind']
    
    for dataset in datasets:
        dataset_raw = raw_dir / dataset
        dataset_processed = processed_dir / dataset
        
        if dataset_raw.exists():
            logger.info(f"Processing {dataset} dataset...")
            results = preprocessor.preprocess_dataset(str(dataset_raw), str(dataset_processed))
            logger.info(f"Completed {dataset} preprocessing")
        else:
            logger.warning(f"Dataset directory not found: {dataset_raw}")

if __name__ == "__main__":
    main()
