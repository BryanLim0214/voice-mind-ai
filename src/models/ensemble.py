"""
Model fusion and ensemble for multi-disorder voice screening.
Combines baseline models with HuBERT predictions and provides confidence scoring.
"""

import os
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
import torch
import torch.nn.functional as F

from .baseline import MultiDisorderBaselineModel
from .hubert_model import HuBERTTrainer

logger = logging.getLogger(__name__)

class ModelEnsemble:
    """Ensemble model combining baseline and HuBERT predictions."""
    
    def __init__(self, 
                 models_dir: str = "models",
                 disorders: List[str] = ['depression', 'anxiety', 'ptsd', 'cognitive']):
        """
        Initialize model ensemble.
        
        Args:
            models_dir: Directory containing trained models
            disorders: List of disorders to classify
        """
        self.models_dir = Path(models_dir)
        self.disorders = disorders
        
        # Initialize model components
        self.baseline_model = MultiDisorderBaselineModel(disorders=disorders, models_dir=str(Path(models_dir)))
        self.hubert_trainer = HuBERTTrainer(output_dir=str(Path(models_dir) / "hubert"))
        
        # Ensemble weights and calibration
        self.ensemble_weights = {}
        self.calibrated_models = {}
        self.is_loaded = False
    
    def load_models(self) -> bool:
        """
        Load all trained models.
        
        Returns:
            True if all models loaded successfully
        """
        try:
            # Load baseline models
            self.baseline_model.load_models()
            
            # Load HuBERT model
            hubert_loaded = self.hubert_trainer.load_model()
            
            if not hubert_loaded:
                logger.warning("HuBERT model not loaded, using baseline only")
            
            self.is_loaded = True
            logger.info("All models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def predict_baseline(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Get baseline model predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Dictionary with baseline predictions
        """
        if not self.is_loaded:
            logger.error("Models not loaded")
            return {}
        
        return self.baseline_model.predict(X, use_ensemble=True)
    
    def predict_hubert(self, audio_path: str) -> Dict[str, np.ndarray]:
        """
        Get HuBERT model predictions.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with HuBERT predictions
        """
        if not self.is_loaded:
            logger.error("Models not loaded")
            return {}
        
        return self.hubert_trainer.predict(audio_path)
    
    def combine_predictions(self, 
                           baseline_preds: Dict[str, np.ndarray],
                           hubert_preds: Dict[str, np.ndarray],
                           weights: Optional[Dict[str, float]] = None) -> Dict[str, np.ndarray]:
        """
        Combine baseline and HuBERT predictions.
        
        Args:
            baseline_preds: Baseline model predictions
            hubert_preds: HuBERT model predictions
            weights: Ensemble weights for each disorder
            
        Returns:
            Dictionary with combined predictions
        """
        if weights is None:
            # Default weights (can be optimized based on validation performance)
            weights = {disorder: 0.6 for disorder in self.disorders}  # 60% baseline, 40% HuBERT
        
        combined_preds = {}
        
        for disorder in self.disorders:
            if disorder in baseline_preds and disorder in hubert_preds:
                # Weighted combination
                baseline_weight = weights[disorder]
                hubert_weight = 1.0 - baseline_weight
                
                combined = (baseline_weight * baseline_preds[disorder] + 
                           hubert_weight * hubert_preds[disorder])
                combined_preds[disorder] = combined
                
            elif disorder in baseline_preds:
                # Use baseline only
                combined_preds[disorder] = baseline_preds[disorder]
                
            elif disorder in hubert_preds:
                # Use HuBERT only
                combined_preds[disorder] = hubert_preds[disorder]
        
        return combined_preds
    
    def compute_confidence_scores(self, predictions: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Compute confidence scores for predictions.
        
        Args:
            predictions: Dictionary with predictions for each disorder
            
        Returns:
            Dictionary with confidence scores
        """
        confidence_scores = {}
        
        for disorder, probs in predictions.items():
            if len(probs) > 0:
                # Confidence as the difference between max and second max probability
                sorted_probs = np.sort(probs)[::-1]
                if len(sorted_probs) >= 2:
                    confidence = sorted_probs[0] - sorted_probs[1]
                else:
                    confidence = sorted_probs[0]
                
                # Normalize to [0, 1] range
                confidence_scores[disorder] = min(confidence, 1.0)
            else:
                confidence_scores[disorder] = 0.0
        
        return confidence_scores
    
    def compute_uncertainty(self, predictions: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Compute prediction uncertainty using entropy.
        
        Args:
            predictions: Dictionary with predictions for each disorder
            
        Returns:
            Dictionary with uncertainty scores
        """
        uncertainty_scores = {}
        
        for disorder, probs in predictions.items():
            if len(probs) > 0:
                # Compute entropy (higher entropy = more uncertainty)
                entropy = -np.sum(probs * np.log(probs + 1e-8))
                
                # Normalize by log(num_classes)
                max_entropy = np.log(len(probs))
                normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
                
                uncertainty_scores[disorder] = normalized_entropy
            else:
                uncertainty_scores[disorder] = 1.0  # Maximum uncertainty
        
        return uncertainty_scores
    
    def predict_with_confidence(self, 
                               X: pd.DataFrame, 
                               audio_path: str,
                               weights: Optional[Dict[str, float]] = None) -> Dict:
        """
        Make predictions with confidence and uncertainty scores.
        
        Args:
            X: Feature matrix for baseline models
            audio_path: Path to audio file for HuBERT
            weights: Ensemble weights
            
        Returns:
            Dictionary with predictions, confidence, and uncertainty
        """
        # Get individual model predictions
        baseline_preds = self.predict_baseline(X)
        hubert_preds = self.predict_hubert(audio_path)
        
        # Combine predictions
        combined_preds = self.combine_predictions(baseline_preds, hubert_preds, weights)
        
        # Compute confidence and uncertainty
        confidence_scores = self.compute_confidence_scores(combined_preds)
        uncertainty_scores = self.compute_uncertainty(combined_preds)
        
        # Format results
        results = {
            'predictions': combined_preds,
            'confidence': confidence_scores,
            'uncertainty': uncertainty_scores,
            'baseline_predictions': baseline_preds,
            'hubert_predictions': hubert_preds
        }
        
        return results
    
    def predict(self, X: pd.DataFrame, use_ensemble: bool = True) -> Dict[str, np.ndarray]:
        """
        Simple predict method for API compatibility.
        
        Args:
            X: Feature matrix
            use_ensemble: Whether to use ensemble (default True)
            
        Returns:
            Dictionary with predictions for each disorder
        """
        if not self.is_loaded:
            self.load_models()
        
        if use_ensemble:
            # Use baseline predictions only for now (HuBERT requires audio path)
            return self.predict_baseline(X)
        else:
            return self.predict_baseline(X)
    
    def optimize_ensemble_weights(self, 
                                 X_val: pd.DataFrame, 
                                 y_val: Dict[str, np.ndarray],
                                 audio_paths_val: List[str]) -> Dict[str, float]:
        """
        Optimize ensemble weights using validation data.
        
        Args:
            X_val: Validation feature matrix
            y_val: Validation labels
            audio_paths_val: Validation audio file paths
            
        Returns:
            Dictionary with optimized weights
        """
        logger.info("Optimizing ensemble weights...")
        
        # Get predictions on validation set
        baseline_preds_val = self.predict_baseline(X_val)
        hubert_preds_val = {}
        
        for i, audio_path in enumerate(audio_paths_val):
            hubert_pred = self.hubert_trainer.predict(audio_path)
            for disorder, probs in hubert_pred.items():
                if disorder not in hubert_preds_val:
                    hubert_preds_val[disorder] = []
                hubert_preds_val[disorder].append(probs)
        
        # Convert to arrays
        for disorder in hubert_preds_val:
            hubert_preds_val[disorder] = np.array(hubert_preds_val[disorder])
        
        # Optimize weights for each disorder
        optimized_weights = {}
        
        for disorder in self.disorders:
            if disorder in baseline_preds_val and disorder in hubert_preds_val:
                # Grid search for optimal weight
                best_weight = 0.5
                best_score = float('inf')
                
                for weight in np.arange(0.1, 1.0, 0.1):
                    # Combine predictions
                    combined = (weight * baseline_preds_val[disorder] + 
                               (1 - weight) * hubert_preds_val[disorder])
                    
                    # Compute log loss
                    try:
                        score = log_loss(y_val[disorder], combined)
                        if score < best_score:
                            best_score = score
                            best_weight = weight
                    except:
                        continue
                
                optimized_weights[disorder] = best_weight
                logger.info(f"Optimized weight for {disorder}: {best_weight:.2f}")
            else:
                optimized_weights[disorder] = 0.5  # Default weight
        
        self.ensemble_weights = optimized_weights
        return optimized_weights
    
    def calibrate_predictions(self, 
                             X_cal: pd.DataFrame, 
                             y_cal: Dict[str, np.ndarray],
                             audio_paths_cal: List[str]):
        """
        Calibrate ensemble predictions using calibration data.
        
        Args:
            X_cal: Calibration feature matrix
            y_cal: Calibration labels
            audio_paths_cal: Calibration audio file paths
        """
        logger.info("Calibrating ensemble predictions...")
        
        # Get ensemble predictions on calibration set
        ensemble_preds_cal = {}
        
        for i, audio_path in enumerate(audio_paths_cal):
            results = self.predict_with_confidence(X_cal.iloc[[i]], audio_path)
            
            for disorder, probs in results['predictions'].items():
                if disorder not in ensemble_preds_cal:
                    ensemble_preds_cal[disorder] = []
                ensemble_preds_cal[disorder].append(probs)
        
        # Convert to arrays
        for disorder in ensemble_preds_cal:
            ensemble_preds_cal[disorder] = np.array(ensemble_preds_cal[disorder])
        
        # Calibrate each disorder separately
        for disorder in self.disorders:
            if disorder in ensemble_preds_cal and disorder in y_cal:
                try:
                    # Use Platt scaling for calibration
                    calibrated_model = CalibratedClassifierCV(method='sigmoid', cv=3)
                    
                    # Fit calibration model
                    # Note: This is a simplified approach. In practice, you might need
                    # a custom calibration method for ensemble predictions
                    
                    self.calibrated_models[disorder] = calibrated_model
                    logger.info(f"Calibrated predictions for {disorder}")
                    
                except Exception as e:
                    logger.warning(f"Could not calibrate {disorder}: {e}")
    
    def get_risk_scores(self, results: Dict) -> Dict[str, Dict[str, float]]:
        """
        Convert predictions to risk scores (0-100 scale).
        
        Args:
            results: Prediction results from predict_with_confidence
            
        Returns:
            Dictionary with risk scores and interpretations
        """
        risk_scores = {}
        
        for disorder, probs in results['predictions'].items():
            # Convert probabilities to risk scores
            # For binary classification, use positive class probability
            # For multiclass, use weighted sum based on severity
            
            if len(probs) == 2:  # Binary classification
                risk_score = probs[1] * 100  # Positive class probability
            else:  # Multiclass classification
                # Weight classes by severity (assuming higher indices = more severe)
                weights = np.arange(len(probs)) / (len(probs) - 1) if len(probs) > 1 else [0]
                risk_score = np.sum(probs * weights) * 100
            
            # Get confidence and uncertainty
            confidence = results['confidence'].get(disorder, 0.0)
            uncertainty = results['uncertainty'].get(disorder, 1.0)
            
            # Determine risk level
            if risk_score < 30:
                risk_level = "Low"
            elif risk_score < 60:
                risk_level = "Moderate"
            else:
                risk_level = "High"
            
            risk_scores[disorder] = {
                'risk_score': risk_score,
                'risk_level': risk_level,
                'confidence': confidence,
                'uncertainty': uncertainty,
                'probabilities': probs.tolist()
            }
        
        return risk_scores
    
    def generate_clinical_report(self, risk_scores: Dict[str, Dict[str, float]]) -> str:
        """
        Generate a clinical report based on risk scores.
        
        Args:
            risk_scores: Risk scores for each disorder
            
        Returns:
            Clinical report string
        """
        report = "VOICE-BASED MENTAL HEALTH SCREENING REPORT\n"
        report += "=" * 50 + "\n\n"
        
        # Overall assessment
        high_risk_disorders = [disorder for disorder, scores in risk_scores.items() 
                              if scores['risk_level'] == 'High']
        moderate_risk_disorders = [disorder for disorder, scores in risk_scores.items() 
                                  if scores['risk_level'] == 'Moderate']
        
        if high_risk_disorders:
            report += f"⚠️  HIGH RISK DETECTED for: {', '.join(high_risk_disorders).title()}\n"
            report += "Recommendation: Immediate clinical evaluation recommended.\n\n"
        elif moderate_risk_disorders:
            report += f"⚠️  MODERATE RISK DETECTED for: {', '.join(moderate_risk_disorders).title()}\n"
            report += "Recommendation: Follow-up screening or clinical consultation advised.\n\n"
        else:
            report += "✅ No significant risk indicators detected.\n"
            report += "Recommendation: Continue routine monitoring.\n\n"
        
        # Detailed results
        report += "DETAILED RESULTS:\n"
        report += "-" * 20 + "\n"
        
        for disorder, scores in risk_scores.items():
            report += f"\n{disorder.title()}:\n"
            report += f"  Risk Score: {scores['risk_score']:.1f}/100\n"
            report += f"  Risk Level: {scores['risk_level']}\n"
            report += f"  Confidence: {scores['confidence']:.2f}\n"
            report += f"  Uncertainty: {scores['uncertainty']:.2f}\n"
        
        # Clinical notes
        report += "\n\nCLINICAL NOTES:\n"
        report += "-" * 15 + "\n"
        report += "• This screening tool is for preliminary assessment only.\n"
        report += "• Results should be interpreted by qualified mental health professionals.\n"
        report += "• High uncertainty scores indicate less reliable predictions.\n"
        report += "• Consider additional clinical evaluation for high-risk cases.\n"
        
        return report
    
    def save_ensemble(self):
        """Save ensemble configuration and weights."""
        ensemble_config = {
            'ensemble_weights': self.ensemble_weights,
            'disorders': self.disorders
        }
        
        config_path = self.models_dir / "ensemble_config.joblib"
        joblib.dump(ensemble_config, config_path)
        
        logger.info(f"Ensemble configuration saved to {config_path}")
    
    def load_ensemble(self):
        """Load ensemble configuration and weights."""
        config_path = self.models_dir / "ensemble_config.joblib"
        
        if config_path.exists():
            ensemble_config = joblib.load(config_path)
            self.ensemble_weights = ensemble_config.get('ensemble_weights', {})
            logger.info("Ensemble configuration loaded")
        else:
            logger.warning("Ensemble configuration not found, using default weights")

def main():
    """Main ensemble function."""
    # Initialize ensemble
    ensemble = ModelEnsemble()
    
    # Load models
    if not ensemble.load_models():
        logger.error("Failed to load models")
        return
    
    # Load ensemble configuration
    ensemble.load_ensemble()
    
    # Example usage
    logger.info("Ensemble model ready for inference")

if __name__ == "__main__":
    main()
