"""
Multi-task HuBERT model for voice-based mental health screening.
Implements transfer learning with task-specific classification heads.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from transformers import (
    HubertModel, 
    Wav2Vec2FeatureExtractor,
    HubertConfig,
    Trainer,
    TrainingArguments
)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import soundfile as sf
import librosa

logger = logging.getLogger(__name__)

class VoiceDataset(Dataset):
    """Dataset class for voice audio data."""
    
    def __init__(self, 
                 audio_paths: List[str], 
                 labels: Dict[str, List], 
                 feature_extractor,
                 max_length: int = 160000):  # 10 seconds at 16kHz
        """
        Initialize dataset.
        
        Args:
            audio_paths: List of audio file paths
            labels: Dictionary with labels for each task
            feature_extractor: Wav2Vec2 feature extractor
            max_length: Maximum audio length in samples
        """
        self.audio_paths = audio_paths
        self.labels = labels
        self.feature_extractor = feature_extractor
        self.max_length = max_length
    
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        # Load audio
        audio_path = self.audio_paths[idx]
        try:
            audio, sr = sf.read(audio_path)
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        except Exception as e:
            logger.error(f"Error loading audio {audio_path}: {e}")
            # Return silence if loading fails
            audio = np.zeros(16000)
        
        # Truncate or pad to max_length
        if len(audio) > self.max_length:
            audio = audio[:self.max_length]
        else:
            audio = np.pad(audio, (0, self.max_length - len(audio)))
        
        # Extract features
        inputs = self.feature_extractor(
            audio, 
            sampling_rate=16000, 
            return_tensors="pt", 
            padding=True
        )
        
        # Get labels for all tasks
        item = {
            'input_values': inputs.input_values.squeeze(0),
            'attention_mask': inputs.attention_mask.squeeze(0)
        }
        
        for task, task_labels in self.labels.items():
            item[f'{task}_labels'] = torch.tensor(task_labels[idx], dtype=torch.long)
        
        return item

class MultiTaskHuBERT(nn.Module):
    """Multi-task HuBERT model for mental health screening."""
    
    def __init__(self, 
                 num_classes: Dict[str, int],
                 model_name: str = "facebook/hubert-base-ls960",
                 freeze_encoder: bool = True,
                 dropout_rate: float = 0.1):
        """
        Initialize multi-task HuBERT model.
        
        Args:
            num_classes: Dictionary with number of classes for each task
            model_name: Pre-trained HuBERT model name
            freeze_encoder: Whether to freeze the HuBERT encoder
            dropout_rate: Dropout rate for classification heads
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.tasks = list(num_classes.keys())
        
        # Load pre-trained HuBERT
        self.hubert = HubertModel.from_pretrained(model_name)
        
        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.hubert.parameters():
                param.requires_grad = False
        
        # Get hidden size
        self.hidden_size = self.hubert.config.hidden_size
        
        # Task-specific classification heads
        self.classifiers = nn.ModuleDict()
        for task, num_class in num_classes.items():
            self.classifiers[task] = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(self.hidden_size, 256),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(256, num_class)
            )
    
    def forward(self, input_values, attention_mask=None, **kwargs):
        """
        Forward pass.
        
        Args:
            input_values: Audio input values
            attention_mask: Attention mask
            **kwargs: Additional arguments (labels for training)
            
        Returns:
            Dictionary with logits for each task
        """
        # Get HuBERT outputs
        outputs = self.hubert(
            input_values=input_values,
            attention_mask=attention_mask
        )
        
        # Use pooled output (mean of last hidden states)
        pooled_output = outputs.last_hidden_state.mean(dim=1)
        
        # Get predictions for each task
        logits = {}
        for task in self.tasks:
            logits[task] = self.classifiers[task](pooled_output)
        
        return logits
    
    def compute_loss(self, logits, labels):
        """
        Compute multi-task loss.
        
        Args:
            logits: Dictionary with logits for each task
            labels: Dictionary with labels for each task
            
        Returns:
            Total loss
        """
        total_loss = 0
        num_tasks = 0
        
        for task in self.tasks:
            if f'{task}_labels' in labels:
                task_logits = logits[task]
                task_labels = labels[f'{task}_labels']
                
                # Compute cross-entropy loss
                loss = F.cross_entropy(task_logits, task_labels)
                total_loss += loss
                num_tasks += 1
        
        return total_loss / num_tasks if num_tasks > 0 else total_loss

class MultiTaskTrainer(Trainer):
    """Custom trainer for multi-task learning."""
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute multi-task loss.
        
        Args:
            model: The model
            inputs: Input batch
            return_outputs: Whether to return outputs
            
        Returns:
            Loss (and optionally outputs)
        """
        # Get logits
        logits = model(
            input_values=inputs['input_values'],
            attention_mask=inputs['attention_mask']
        )
        
        # Compute loss
        loss = model.compute_loss(logits, inputs)
        
        return (loss, logits) if return_outputs else loss

class HuBERTTrainer:
    """Trainer for multi-task HuBERT model."""
    
    def __init__(self, 
                 model_name: str = "facebook/hubert-base-ls960",
                 output_dir: str = "models/hubert",
                 freeze_encoder: bool = True):
        """
        Initialize HuBERT trainer.
        
        Args:
            model_name: Pre-trained model name
            output_dir: Output directory for models
            freeze_encoder: Whether to freeze encoder
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.freeze_encoder = freeze_encoder
        
        # Initialize feature extractor
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        
        # Model and trainer will be initialized during training
        self.model = None
        self.trainer = None
        self.label_encoders = {}
    
    def prepare_data(self, features_df: pd.DataFrame) -> Tuple[List[str], Dict[str, List]]:
        """
        Prepare data for training.
        
        Args:
            features_df: DataFrame with features and metadata
            
        Returns:
            Tuple of (audio_paths, labels_dict)
        """
        # Get audio paths
        audio_paths = []
        labels = {
            'depression': [],
            'anxiety': [],
            'ptsd': [],
            'cognitive': []
        }
        
        for idx, row in features_df.iterrows():
            participant_id = row['participant_id']
            dataset = row['dataset']
            
            # Construct audio file path
            audio_path = f"data/processed/{dataset}/{participant_id}_processed.wav"
            
            if os.path.exists(audio_path):
                audio_paths.append(audio_path)
                
                # Get labels
                for task in labels.keys():
                    label_col = f"{task}_label"
                    if label_col in row:
                        labels[task].append(row[label_col])
                    else:
                        labels[task].append('none')  # Default label
        
        # Encode labels
        for task in labels.keys():
            le = LabelEncoder()
            labels[task] = le.fit_transform(labels[task])
            self.label_encoders[task] = le
        
        return audio_paths, labels
    
    def create_datasets(self, audio_paths: List[str], labels: Dict[str, List], 
                       test_size: float = 0.2) -> Tuple[Dataset, Dataset]:
        """
        Create train and validation datasets.
        
        Args:
            audio_paths: List of audio file paths
            labels: Dictionary with labels
            test_size: Fraction of data for validation
            
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        # Split data
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            audio_paths, labels, test_size=test_size, random_state=42, stratify=labels['depression']
        )
        
        # Create datasets
        train_dataset = VoiceDataset(train_paths, train_labels, self.feature_extractor)
        val_dataset = VoiceDataset(val_paths, val_labels, self.feature_extractor)
        
        return train_dataset, val_dataset
    
    def train(self, features_df: pd.DataFrame, 
              num_epochs: int = 10,
              batch_size: int = 4,
              learning_rate: float = 1e-4,
              gradient_accumulation_steps: int = 8):
        """
        Train the multi-task HuBERT model.
        
        Args:
            features_df: DataFrame with features and metadata
            num_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            gradient_accumulation_steps: Gradient accumulation steps
        """
        # Prepare data
        audio_paths, labels = self.prepare_data(features_df)
        
        if len(audio_paths) == 0:
            logger.error("No audio files found for training")
            return
        
        # Create datasets
        train_dataset, val_dataset = self.create_datasets(audio_paths, labels)
        
        # Get number of classes for each task
        num_classes = {}
        for task, task_labels in labels.items():
            num_classes[task] = len(np.unique(task_labels))
        
        # Initialize model
        self.model = MultiTaskHuBERT(
            num_classes=num_classes,
            model_name=self.model_name,
            freeze_encoder=self.freeze_encoder
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=100,
            logging_steps=10,
            eval_steps=100,
            save_steps=500,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=None,  # Disable wandb/tensorboard
            dataloader_pin_memory=False,  # Reduce memory usage
        )
        
        # Initialize trainer
        self.trainer = MultiTaskTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )
        
        # Train model
        logger.info("Starting HuBERT training...")
        self.trainer.train()
        
        # Save model
        self.save_model()
        
        logger.info("HuBERT training completed")
    
    def save_model(self):
        """Save the trained model and components."""
        if self.model is None:
            logger.error("No model to save")
            return
        
        # Save model
        model_path = self.output_dir / "model"
        self.model.save_pretrained(model_path)
        
        # Save feature extractor
        feature_extractor_path = self.output_dir / "feature_extractor"
        self.feature_extractor.save_pretrained(feature_extractor_path)
        
        # Save label encoders
        import joblib
        encoders_path = self.output_dir / "label_encoders.joblib"
        joblib.dump(self.label_encoders, encoders_path)
        
        logger.info(f"Model saved to {self.output_dir}")
    
    def load_model(self):
        """Load pre-trained model and components."""
        model_path = self.output_dir / "model"
        feature_extractor_path = self.output_dir / "feature_extractor"
        encoders_path = self.output_dir / "label_encoders.joblib"
        
        if not all([model_path.exists(), feature_extractor_path.exists(), encoders_path.exists()]):
            logger.error("Model files not found")
            return False
        
        # Load feature extractor
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(feature_extractor_path)
        
        # Load label encoders
        import joblib
        self.label_encoders = joblib.load(encoders_path)
        
        # Load model
        self.model = MultiTaskHuBERT.from_pretrained(model_path)
        self.model.eval()
        
        logger.info("Model loaded successfully")
        return True
    
    def predict(self, audio_path: str) -> Dict[str, np.ndarray]:
        """
        Make predictions on a single audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with predictions for each task
        """
        if self.model is None:
            logger.error("Model not loaded")
            return {}
        
        # Load and preprocess audio
        try:
            audio, sr = sf.read(audio_path)
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        except Exception as e:
            logger.error(f"Error loading audio {audio_path}: {e}")
            return {}
        
        # Truncate or pad to 10 seconds
        max_length = 160000  # 10 seconds at 16kHz
        if len(audio) > max_length:
            audio = audio[:max_length]
        else:
            audio = np.pad(audio, (0, max_length - len(audio)))
        
        # Extract features
        inputs = self.feature_extractor(
            audio, 
            sampling_rate=16000, 
            return_tensors="pt"
        )
        
        # Move to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Make prediction
        with torch.no_grad():
            logits = self.model(**inputs)
        
        # Convert to probabilities
        predictions = {}
        for task, task_logits in logits.items():
            probs = F.softmax(task_logits, dim=-1)
            predictions[task] = probs.cpu().numpy()[0]
        
        return predictions
    
    def evaluate(self, features_df: pd.DataFrame) -> Dict:
        """
        Evaluate model performance.
        
        Args:
            features_df: DataFrame with test data
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            logger.error("Model not loaded")
            return {}
        
        # Prepare test data
        audio_paths, labels = self.prepare_data(features_df)
        
        if len(audio_paths) == 0:
            logger.error("No test data found")
            return {}
        
        # Make predictions
        all_predictions = {}
        for task in self.label_encoders.keys():
            all_predictions[task] = []
        
        for audio_path in audio_paths:
            predictions = self.predict(audio_path)
            for task, probs in predictions.items():
                all_predictions[task].append(probs)
        
        # Calculate metrics
        metrics = {}
        for task in self.label_encoders.keys():
            if task in all_predictions and len(all_predictions[task]) > 0:
                y_pred_proba = np.array(all_predictions[task])
                y_pred = np.argmax(y_pred_proba, axis=1)
                y_true = labels[task]
                
                # Calculate accuracy
                accuracy = np.mean(y_pred == y_true)
                
                # Calculate AUC (if binary or multiclass)
                try:
                    from sklearn.metrics import roc_auc_score
                    if len(np.unique(y_true)) == 2:
                        auc = roc_auc_score(y_true, y_pred_proba[:, 1])
                    else:
                        auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
                except:
                    auc = 0.0
                
                metrics[task] = {
                    'accuracy': accuracy,
                    'auc': auc
                }
        
        return metrics

def main():
    """Main training function."""
    # Load features
    features_path = "data/features/extracted_features.csv"
    if not os.path.exists(features_path):
        logger.error(f"Features file not found: {features_path}")
        return
    
    features_df = pd.read_csv(features_path)
    logger.info(f"Loaded features: {features_df.shape}")
    
    # Initialize trainer
    trainer = HuBERTTrainer()
    
    # Train model
    trainer.train(features_df, num_epochs=5, batch_size=2)  # Reduced for local training
    
    # Evaluate model
    metrics = trainer.evaluate(features_df)
    logger.info("Evaluation results:")
    for task, task_metrics in metrics.items():
        logger.info(f"{task}: Accuracy = {task_metrics['accuracy']:.3f}, AUC = {task_metrics['auc']:.3f}")

if __name__ == "__main__":
    main()
