"""
Data loading utilities for multi-disorder voice screening datasets.
Handles loading and organizing data from different sources.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import json

logger = logging.getLogger(__name__)

class DatasetLoader:
    """Handles loading and organizing voice screening datasets."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize dataset loader.
        
        Args:
            data_dir: Base data directory path
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
    def load_voiceome_metadata(self) -> pd.DataFrame:
        """
        Load Voiceome dataset metadata.
        
        Returns:
            DataFrame with participant metadata and labels
        """
        # This would load the actual Voiceome metadata
        # For now, creating a template structure
        metadata = {
            'participant_id': [],
            'age': [],
            'gender': [],
            'depression_severity': [],
            'anxiety_severity': [],
            'ptsd_present': [],
            'cognitive_status': [],
            'audio_file': []
        }
        
        # In practice, this would read from the actual Voiceome files
        # For demonstration, creating sample data
        sample_data = {
            'participant_id': [f'V{i:04d}' for i in range(100)],
            'age': np.random.randint(18, 80, 100),
            'gender': np.random.choice(['M', 'F'], 100),
            'depression_severity': np.random.choice(['none', 'mild', 'moderate', 'severe'], 100, p=[0.6, 0.2, 0.15, 0.05]),
            'anxiety_severity': np.random.choice(['none', 'mild', 'moderate', 'severe'], 100, p=[0.5, 0.3, 0.15, 0.05]),
            'ptsd_present': np.random.choice(['none', 'present'], 100, p=[0.8, 0.2]),
            'cognitive_status': np.random.choice(['normal', 'mild_impairment', 'moderate_impairment'], 100, p=[0.7, 0.25, 0.05]),
            'audio_file': [f'V{i:04d}_voice.wav' for i in range(100)]
        }
        
        return pd.DataFrame(sample_data)
    
    def load_daic_metadata(self) -> pd.DataFrame:
        """
        Load DAIC-WOZ dataset metadata.
        
        Returns:
            DataFrame with participant metadata and labels
        """
        # DAIC-WOZ structure
        sample_data = {
            'participant_id': [f'D{i:03d}' for i in range(50)],
            'age': np.random.randint(18, 65, 50),
            'gender': np.random.choice(['M', 'F'], 50),
            'depression_severity': np.random.choice(['none', 'mild', 'moderate', 'severe'], 50, p=[0.4, 0.3, 0.2, 0.1]),
            'anxiety_severity': np.random.choice(['none', 'mild', 'moderate', 'severe'], 50, p=[0.5, 0.3, 0.15, 0.05]),
            'ptsd_present': np.random.choice(['none', 'present'], 50, p=[0.7, 0.3]),
            'audio_file': [f'D{i:03d}_interview.wav' for i in range(50)]
        }
        
        return pd.DataFrame(sample_data)
    
    def load_vocal_mind_metadata(self) -> pd.DataFrame:
        """
        Load Vocal Mind dataset metadata.
        
        Returns:
            DataFrame with participant metadata and MADRS scores
        """
        # Vocal Mind with MADRS scores
        sample_data = {
            'participant_id': [f'VM{i:03d}' for i in range(30)],
            'age': np.random.randint(18, 75, 30),
            'gender': np.random.choice(['M', 'F'], 30),
            'madrs_score': np.random.randint(0, 60, 30),
            'depression_severity': [],
            'audio_file': [f'VM{i:03d}_speech.wav' for i in range(30)]
        }
        
        # Convert MADRS scores to severity categories
        for score in sample_data['madrs_score']:
            if score < 7:
                sample_data['depression_severity'].append('none')
            elif score < 20:
                sample_data['depression_severity'].append('mild')
            elif score < 35:
                sample_data['depression_severity'].append('moderate')
            else:
                sample_data['depression_severity'].append('severe')
        
        return pd.DataFrame(sample_data)
    
    def create_unified_dataset(self) -> pd.DataFrame:
        """
        Create unified dataset combining all sources.
        
        Returns:
            Combined DataFrame with all participants
        """
        datasets = []
        
        # Load each dataset
        try:
            voiceome_df = self.load_voiceome_metadata()
            voiceome_df['dataset'] = 'voiceome'
            datasets.append(voiceome_df)
        except Exception as e:
            logger.warning(f"Could not load Voiceome: {e}")
        
        try:
            daic_df = self.load_daic_metadata()
            daic_df['dataset'] = 'daic'
            datasets.append(daic_df)
        except Exception as e:
            logger.warning(f"Could not load DAIC: {e}")
        
        try:
            vocal_mind_df = self.load_vocal_mind_metadata()
            vocal_mind_df['dataset'] = 'vocal_mind'
            datasets.append(vocal_mind_df)
        except Exception as e:
            logger.warning(f"Could not load Vocal Mind: {e}")
        
        if not datasets:
            raise ValueError("No datasets could be loaded")
        
        # Combine datasets
        combined_df = pd.concat(datasets, ignore_index=True)
        
        # Fill missing columns with default values
        default_columns = {
            'depression_severity': 'none',
            'anxiety_severity': 'none', 
            'ptsd_present': 'none',
            'cognitive_status': 'normal',
            'madrs_score': 0
        }
        
        for col, default_val in default_columns.items():
            if col not in combined_df.columns:
                combined_df[col] = default_val
            else:
                combined_df[col] = combined_df[col].fillna(default_val)
        
        return combined_df
    
    def create_label_encoding(self, df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
        """
        Create label encoding for multi-disorder classification.
        
        Args:
            df: Combined dataset DataFrame
            
        Returns:
            Dictionary with label encodings for each disorder
        """
        encodings = {}
        
        # Depression encoding
        depression_labels = df['depression_severity'].unique()
        encodings['depression'] = {label: i for i, label in enumerate(sorted(depression_labels))}
        
        # Anxiety encoding
        anxiety_labels = df['anxiety_severity'].unique()
        encodings['anxiety'] = {label: i for i, label in enumerate(sorted(anxiety_labels))}
        
        # PTSD encoding
        ptsd_labels = df['ptsd_present'].unique()
        encodings['ptsd'] = {label: i for i, label in enumerate(sorted(ptsd_labels))}
        
        # Cognitive encoding
        cognitive_labels = df['cognitive_status'].unique()
        encodings['cognitive'] = {label: i for i, label in enumerate(sorted(cognitive_labels))}
        
        return encodings
    
    def save_metadata(self, df: pd.DataFrame, encodings: Dict[str, Dict[str, int]]):
        """
        Save processed metadata and encodings.
        
        Args:
            df: Combined dataset DataFrame
            encodings: Label encodings dictionary
        """
        # Save metadata
        metadata_path = self.processed_dir / "unified_metadata.csv"
        df.to_csv(metadata_path, index=False)
        
        # Save encodings
        encodings_path = self.processed_dir / "label_encodings.json"
        with open(encodings_path, 'w') as f:
            json.dump(encodings, f, indent=2)
        
        logger.info(f"Saved metadata to {metadata_path}")
        logger.info(f"Saved encodings to {encodings_path}")
    
    def get_audio_file_path(self, participant_id: str, dataset: str) -> str:
        """
        Get the path to an audio file for a participant.
        
        Args:
            participant_id: Participant identifier
            dataset: Dataset name
            
        Returns:
            Path to audio file
        """
        return str(self.processed_dir / dataset / f"{participant_id}_processed.wav")

def main():
    """Main data loading function."""
    loader = DatasetLoader()
    
    # Create unified dataset
    logger.info("Creating unified dataset...")
    unified_df = loader.create_unified_dataset()
    
    # Create label encodings
    logger.info("Creating label encodings...")
    encodings = loader.create_label_encoding(unified_df)
    
    # Save processed data
    logger.info("Saving processed data...")
    loader.save_metadata(unified_df, encodings)
    
    logger.info(f"Unified dataset created with {len(unified_df)} participants")
    logger.info(f"Datasets included: {unified_df['dataset'].unique()}")

if __name__ == "__main__":
    main()
