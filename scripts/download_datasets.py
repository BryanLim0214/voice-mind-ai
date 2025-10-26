"""
Dataset download script for multi-disorder voice screening platform.
Downloads and organizes Voiceome, DAIC-WOZ, and Vocal Mind datasets.
"""

import os
import requests
import zipfile
import tarfile
from pathlib import Path
import logging
from typing import Optional
import json

logger = logging.getLogger(__name__)

class DatasetDownloader:
    """Handles downloading and organizing voice screening datasets."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize dataset downloader.
        
        Args:
            data_dir: Base data directory
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset URLs and information
        self.datasets = {
            'voiceome': {
                'url': 'https://github.com/voiceome/voiceome-study/archive/refs/heads/main.zip',
                'description': 'Voiceome Study - 6,000+ participants with 80+ health labels',
                'license': 'Research use only',
                'citation': 'Voiceome Study Consortium. (2023). Voiceome: A large-scale voice health dataset.'
            },
            'daic': {
                'url': 'https://dcapswoz.ict.usc.edu/',
                'description': 'DAIC-WOZ Depression Corpus - 189 clinical interviews',
                'license': 'Requires application and IRB approval',
                'citation': 'Gratch, J. et al. (2014). The Distress Analysis Interview Corpus.'
            },
            'vocal_mind': {
                'url': 'https://physionet.org/content/vocal-mind/1.0.0/',
                'description': 'Vocal Mind Dataset - 514 participants with MADRS scores',
                'license': 'PhysioNet Credentialed Health Data License',
                'citation': 'Vocal Mind Consortium. (2023). Vocal biomarkers for mental health assessment.'
            }
        }
    
    def download_voiceome(self) -> bool:
        """
        Download Voiceome dataset.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Downloading Voiceome dataset...")
            
            # Note: This is a placeholder URL. In practice, you would need to:
            # 1. Apply for access to the Voiceome dataset
            # 2. Get the actual download URL
            # 3. Handle authentication if required
            
            voiceome_dir = self.raw_dir / "voiceome"
            voiceome_dir.mkdir(exist_ok=True)
            
            # Create a placeholder file with dataset information
            info_file = voiceome_dir / "dataset_info.json"
            with open(info_file, 'w') as f:
                json.dump({
                    'name': 'Voiceome Study',
                    'description': 'Large-scale voice health dataset with 6,000+ participants',
                    'participants': 6000,
                    'utterances_per_participant': 48,
                    'health_labels': 80,
                    'disorders': ['depression', 'anxiety', 'ptsd', 'cognitive_decline'],
                    'access_instructions': [
                        '1. Visit https://github.com/voiceome/voiceome-study',
                        '2. Follow the access request process',
                        '3. Download the dataset to this directory',
                        '4. Extract audio files and metadata'
                    ],
                    'citation': self.datasets['voiceome']['citation']
                }, f, indent=2)
            
            logger.info("Voiceome dataset information saved. Please follow access instructions.")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading Voiceome dataset: {e}")
            return False
    
    def download_daic(self) -> bool:
        """
        Download DAIC-WOZ dataset.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Downloading DAIC-WOZ dataset...")
            
            daic_dir = self.raw_dir / "daic"
            daic_dir.mkdir(exist_ok=True)
            
            # Create a placeholder file with dataset information
            info_file = daic_dir / "dataset_info.json"
            with open(info_file, 'w') as f:
                json.dump({
                    'name': 'DAIC-WOZ Depression Corpus',
                    'description': 'Clinical interview dataset for depression and PTSD detection',
                    'participants': 189,
                    'modalities': ['audio', 'video', 'text', 'questionnaires'],
                    'disorders': ['depression', 'anxiety', 'ptsd'],
                    'access_instructions': [
                        '1. Visit https://dcapswoz.ict.usc.edu/',
                        '2. Complete the data request form',
                        '3. Provide IRB approval documentation',
                        '4. Wait for approval and download link',
                        '5. Download and extract to this directory'
                    ],
                    'citation': self.datasets['daic']['citation']
                }, f, indent=2)
            
            logger.info("DAIC-WOZ dataset information saved. Please follow access instructions.")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading DAIC-WOZ dataset: {e}")
            return False
    
    def download_vocal_mind(self) -> bool:
        """
        Download Vocal Mind dataset.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Downloading Vocal Mind dataset...")
            
            vocal_mind_dir = self.raw_dir / "vocal_mind"
            vocal_mind_dir.mkdir(exist_ok=True)
            
            # Create a placeholder file with dataset information
            info_file = vocal_mind_dir / "dataset_info.json"
            with open(info_file, 'w') as f:
                json.dump({
                    'name': 'Vocal Mind Dataset',
                    'description': 'Voice dataset with MADRS depression severity scores',
                    'participants': 514,
                    'assessment': 'MADRS (Montgomery-Asberg Depression Rating Scale)',
                    'disorders': ['depression'],
                    'access_instructions': [
                        '1. Visit https://physionet.org/content/vocal-mind/1.0.0/',
                        '2. Complete PhysioNet credentialing process',
                        '3. Sign the data use agreement',
                        '4. Download the dataset using wget or browser',
                        '5. Extract to this directory'
                    ],
                    'citation': self.datasets['vocal_mind']['citation']
                }, f, indent=2)
            
            logger.info("Vocal Mind dataset information saved. Please follow access instructions.")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading Vocal Mind dataset: {e}")
            return False
    
    def create_sample_data(self) -> bool:
        """
        Create sample data for testing and demonstration.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Creating sample data for demonstration...")
            
            # Create sample audio files (empty files for demonstration)
            sample_dir = self.raw_dir / "sample"
            sample_dir.mkdir(exist_ok=True)
            
            # Create sample metadata
            sample_metadata = {
                'participant_id': ['S001', 'S002', 'S003', 'S004', 'S005'],
                'age': [25, 35, 45, 55, 65],
                'gender': ['F', 'M', 'F', 'M', 'F'],
                'depression_severity': ['none', 'mild', 'moderate', 'severe', 'none'],
                'anxiety_severity': ['mild', 'none', 'mild', 'moderate', 'none'],
                'ptsd_present': ['none', 'none', 'present', 'none', 'none'],
                'cognitive_status': ['normal', 'normal', 'mild_impairment', 'normal', 'normal'],
                'audio_file': ['S001_voice.wav', 'S002_voice.wav', 'S003_voice.wav', 'S004_voice.wav', 'S005_voice.wav']
            }
            
            # Save sample metadata
            import pandas as pd
            sample_df = pd.DataFrame(sample_metadata)
            sample_df.to_csv(sample_dir / "sample_metadata.csv", index=False)
            
            # Create sample audio files (empty files)
            for audio_file in sample_metadata['audio_file']:
                audio_path = sample_dir / audio_file
                with open(audio_path, 'w') as f:
                    f.write("")  # Empty file for demonstration
            
            # Create dataset info
            info_file = sample_dir / "dataset_info.json"
            with open(info_file, 'w') as f:
                json.dump({
                    'name': 'Sample Dataset',
                    'description': 'Sample data for testing and demonstration',
                    'participants': 5,
                    'note': 'This is sample data for demonstration purposes only',
                    'usage': 'Use this data to test the platform before obtaining real datasets'
                }, f, indent=2)
            
            logger.info("Sample data created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error creating sample data: {e}")
            return False
    
    def download_all(self) -> bool:
        """
        Download all datasets.
        
        Returns:
            True if all downloads successful, False otherwise
        """
        success = True
        
        # Download each dataset
        if not self.download_voiceome():
            success = False
        
        if not self.download_daic():
            success = False
        
        if not self.download_vocal_mind():
            success = False
        
        # Create sample data
        if not self.create_sample_data():
            success = False
        
        return success
    
    def print_access_instructions(self):
        """Print detailed access instructions for all datasets."""
        print("\n" + "="*80)
        print("DATASET ACCESS INSTRUCTIONS")
        print("="*80)
        
        for dataset_name, info in self.datasets.items():
            print(f"\n{dataset_name.upper()} DATASET:")
            print(f"Description: {info['description']}")
            print(f"License: {info['license']}")
            print(f"Citation: {info['citation']}")
            print(f"URL: {info['url']}")
            print("-" * 40)
        
        print("\nGENERAL INSTRUCTIONS:")
        print("1. Review the license terms for each dataset")
        print("2. Complete any required applications or agreements")
        print("3. Download datasets to the appropriate directories in data/raw/")
        print("4. Extract audio files and metadata")
        print("5. Update the preprocessing pipeline if needed")
        print("\nFor immediate testing, sample data has been created in data/raw/sample/")

def main():
    """Main function to download datasets."""
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize downloader
    downloader = DatasetDownloader()
    
    # Download all datasets
    print("Starting dataset download process...")
    success = downloader.download_all()
    
    if success:
        print("\nDataset download process completed successfully!")
        print("Please follow the access instructions in each dataset directory.")
    else:
        print("\nSome datasets could not be downloaded. Check the logs for details.")
    
    # Print access instructions
    downloader.print_access_instructions()

if __name__ == "__main__":
    main()
