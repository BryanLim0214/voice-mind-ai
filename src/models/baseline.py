"""
Baseline ensemble models for multi-disorder voice screening.
Implements Random Forest, XGBoost, and SVM with cross-validation.
"""

import os
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb

import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class MultiDisorderBaselineModel:
    """Baseline ensemble models for multi-disorder classification."""
    
    def __init__(self, 
                 disorders: List[str] = ['depression', 'anxiety', 'ptsd', 'cognitive'],
                 models_dir: str = "models"):
        """
        Initialize baseline model trainer.
        
        Args:
            disorders: List of disorders to classify
            models_dir: Directory to save trained models
        """
        self.disorders = disorders
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Initialize models
        self.models = {}
        self.label_encoders = {}
        self.scalers = {}
        self.ensemble_models = {}
        
        # Model configurations
        self.model_configs = {
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42
            },
            'xgboost': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            },
            'svm': {
                'C': 1.0,
                'kernel': 'rbf',
                'gamma': 'scale',
                'random_state': 42
            }
        }
    
    def prepare_data(self, features_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
        """
        Prepare data for training.
        
        Args:
            features_df: DataFrame with features and labels
            
        Returns:
            Tuple of (features, labels_dict)
        """
        # Get feature columns (exclude metadata and labels)
        exclude_cols = ['participant_id', 'dataset', 'age', 'gender', 
                       'depression_label', 'anxiety_label', 'ptsd_label', 'cognitive_label']
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        
        X = features_df[feature_cols].fillna(0)
        
        # Prepare labels for each disorder
        y_dict = {}
        for disorder in self.disorders:
            label_col = f"{disorder}_label"
            if label_col in features_df.columns:
                # Encode labels
                le = LabelEncoder()
                y_dict[disorder] = le.fit_transform(features_df[label_col])
                self.label_encoders[disorder] = le
            else:
                logger.warning(f"Label column {label_col} not found")
        
        return X, y_dict
    
    def train_individual_models(self, X: pd.DataFrame, y: np.ndarray, disorder: str) -> Dict:
        """
        Train individual models for a specific disorder.
        
        Args:
            X: Feature matrix
            y: Labels
            disorder: Disorder name
            
        Returns:
            Dictionary with trained models
        """
        models = {}
        
        # Scale features for SVM
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers[disorder] = scaler
        
        # Random Forest
        logger.info(f"Training Random Forest for {disorder}")
        rf_model = RandomForestClassifier(**self.model_configs['random_forest'])
        rf_model.fit(X, y)
        models['random_forest'] = rf_model
        
        # XGBoost
        logger.info(f"Training XGBoost for {disorder}")
        xgb_model = xgb.XGBClassifier(**self.model_configs['xgboost'])
        xgb_model.fit(X, y)
        models['xgboost'] = xgb_model
        
        # SVM
        logger.info(f"Training SVM for {disorder}")
        svm_model = SVC(**self.model_configs['svm'], probability=True)
        svm_model.fit(X_scaled, y)
        models['svm'] = svm_model
        
        return models
    
    def create_ensemble_model(self, models: Dict, X: pd.DataFrame, y: np.ndarray, disorder: str) -> VotingClassifier:
        """
        Create ensemble model using voting classifier.
        
        Args:
            models: Dictionary with individual models
            X: Feature matrix
            y: Labels
            disorder: Disorder name
            
        Returns:
            Voting classifier
        """
        # Create voting classifier
        estimators = [(name, model) for name, model in models.items()]
        ensemble = VotingClassifier(estimators=estimators, voting='soft')
        
        # Train ensemble
        logger.info(f"Training ensemble model for {disorder}")
        if 'svm' in models:
            # Use scaled features for ensemble (SVM needs scaling)
            X_scaled = self.scalers[disorder].transform(X)
            ensemble.fit(X_scaled, y)
        else:
            ensemble.fit(X, y)
        
        return ensemble
    
    def evaluate_model(self, model, X: pd.DataFrame, y: np.ndarray, disorder: str, 
                      cv_folds: int = 5) -> Dict:
        """
        Evaluate model performance using cross-validation.
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Labels
            disorder: Disorder name
            cv_folds: Number of CV folds
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Use scaled features if model is SVM or ensemble
        if hasattr(model, 'estimators_') and any('svm' in str(est) for est in model.estimators_):
            X_eval = self.scalers[disorder].transform(X)
        elif hasattr(model, 'kernel'):  # SVM
            X_eval = self.scalers[disorder].transform(X)
        else:
            X_eval = X
        
        # Cross-validation scores
        cv_scores = cross_val_score(model, X_eval, y, cv=cv, scoring='roc_auc_ovr')
        
        # Full dataset predictions for additional metrics
        y_pred = model.predict(X_eval)
        y_pred_proba = model.predict_proba(X_eval)
        
        # Calculate metrics
        metrics = {
            'cv_auc_mean': np.mean(cv_scores),
            'cv_auc_std': np.std(cv_scores),
            'cv_scores': cv_scores.tolist()
        }
        
        # Classification report
        try:
            report = classification_report(y, y_pred, output_dict=True)
            metrics['classification_report'] = report
        except:
            logger.warning(f"Could not generate classification report for {disorder}")
        
        # Confusion matrix
        try:
            cm = confusion_matrix(y, y_pred)
            metrics['confusion_matrix'] = cm.tolist()
        except:
            logger.warning(f"Could not generate confusion matrix for {disorder}")
        
        return metrics
    
    def train_all_models(self, features_df: pd.DataFrame) -> Dict:
        """
        Train models for all disorders.
        
        Args:
            features_df: DataFrame with features and labels
            
        Returns:
            Dictionary with training results
        """
        # Prepare data
        X, y_dict = self.prepare_data(features_df)
        
        results = {}
        
        for disorder in self.disorders:
            if disorder not in y_dict:
                logger.warning(f"Skipping {disorder} - no labels found")
                continue
            
            y = y_dict[disorder]
            logger.info(f"Training models for {disorder} with {len(X)} samples")
            
            # Train individual models
            individual_models = self.train_individual_models(X, y, disorder)
            self.models[disorder] = individual_models
            
            # Create ensemble
            ensemble_model = self.create_ensemble_model(individual_models, X, y, disorder)
            self.ensemble_models[disorder] = ensemble_model
            
            # Evaluate models
            disorder_results = {}
            
            # Evaluate individual models
            for model_name, model in individual_models.items():
                logger.info(f"Evaluating {model_name} for {disorder}")
                metrics = self.evaluate_model(model, X, y, disorder)
                disorder_results[model_name] = metrics
            
            # Evaluate ensemble
            logger.info(f"Evaluating ensemble for {disorder}")
            ensemble_metrics = self.evaluate_model(ensemble_model, X, y, disorder)
            disorder_results['ensemble'] = ensemble_metrics
            
            results[disorder] = disorder_results
        
        return results
    
    def save_models(self):
        """Save all trained models."""
        # Save individual models
        for disorder, models in self.models.items():
            disorder_dir = self.models_dir / disorder
            disorder_dir.mkdir(exist_ok=True)
            
            for model_name, model in models.items():
                model_path = disorder_dir / f"{model_name}.joblib"
                joblib.dump(model, model_path)
                logger.info(f"Saved {model_name} for {disorder} to {model_path}")
        
        # Save ensemble models
        for disorder, ensemble in self.ensemble_models.items():
            ensemble_path = self.models_dir / f"{disorder}_ensemble.joblib"
            joblib.dump(ensemble, ensemble_path)
            logger.info(f"Saved ensemble for {disorder} to {ensemble_path}")
        
        # Save scalers and encoders
        scalers_path = self.models_dir / "scalers.joblib"
        joblib.dump(self.scalers, scalers_path)
        
        encoders_path = self.models_dir / "label_encoders.joblib"
        joblib.dump(self.label_encoders, encoders_path)
        
        logger.info("All models saved successfully")
    
    def load_models(self):
        """Load pre-trained models."""
        # Load individual models
        for disorder in self.disorders:
            disorder_dir = Path(self.models_dir) / disorder
            if disorder_dir.exists():
                self.models[disorder] = {}
                for model_file in disorder_dir.glob("*.joblib"):
                    model_name = model_file.stem
                    model = joblib.load(model_file)
                    self.models[disorder][model_name] = model
        
        # Load ensemble models
        for disorder in self.disorders:
            ensemble_path = Path(self.models_dir) / f"{disorder}_ensemble.joblib"
            if ensemble_path.exists():
                self.ensemble_models[disorder] = joblib.load(ensemble_path)
        
        # Load scalers and encoders
        scalers_path = Path(self.models_dir) / "scalers.joblib"
        if scalers_path.exists():
            self.scalers = joblib.load(scalers_path)
        
        encoders_path = Path(self.models_dir) / "label_encoders.joblib"
        if encoders_path.exists():
            self.label_encoders = joblib.load(encoders_path)
        
        logger.info("Models loaded successfully")
    
    def predict(self, X: pd.DataFrame, use_ensemble: bool = True) -> Dict[str, np.ndarray]:
        """
        Make predictions for all disorders.
        
        Args:
            X: Feature matrix
            use_ensemble: Whether to use ensemble models
            
        Returns:
            Dictionary with predictions for each disorder
        """
        predictions = {}
        
        for disorder in self.disorders:
            if use_ensemble and disorder in self.ensemble_models:
                model = self.ensemble_models[disorder]
            elif disorder in self.models:
                # Use best individual model (Random Forest by default)
                model = self.models[disorder]['random_forest']
            else:
                logger.warning(f"No model found for {disorder}")
                continue
            
            # Scale features if needed
            if hasattr(model, 'estimators_') and any('svm' in str(est) for est in model.estimators_):
                X_pred = self.scalers[disorder].transform(X)
            elif hasattr(model, 'kernel'):  # SVM
                X_pred = self.scalers[disorder].transform(X)
            else:
                X_pred = X
            
            # Make predictions
            y_pred_proba = model.predict_proba(X_pred)
            predictions[disorder] = y_pred_proba
        
        return predictions
    
    def get_feature_importance(self, disorder: str, model_name: str = 'random_forest') -> pd.DataFrame:
        """
        Get feature importance for a specific model.
        
        Args:
            disorder: Disorder name
            model_name: Model name
            
        Returns:
            DataFrame with feature importance
        """
        if disorder not in self.models or model_name not in self.models[disorder]:
            logger.error(f"Model {model_name} not found for {disorder}")
            return pd.DataFrame()
        
        model = self.models[disorder][model_name]
        
        if hasattr(model, 'feature_importances_'):
            # Get feature names (this would need to be passed from the original data)
            # For now, return importance values
            importance_df = pd.DataFrame({
                'feature': range(len(model.feature_importances_)),
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        else:
            logger.warning(f"Model {model_name} does not have feature importance")
            return pd.DataFrame()
    
    def plot_results(self, results: Dict, output_dir: str = "results"):
        """
        Plot training results.
        
        Args:
            results: Training results dictionary
            output_dir: Output directory for plots
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Plot CV scores for each disorder
        for disorder, disorder_results in results.items():
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Model Performance for {disorder.title()}', fontsize=16)
            
            # CV AUC scores
            model_names = list(disorder_results.keys())
            cv_means = [disorder_results[model]['cv_auc_mean'] for model in model_names]
            cv_stds = [disorder_results[model]['cv_auc_std'] for model in model_names]
            
            axes[0, 0].bar(model_names, cv_means, yerr=cv_stds, capsize=5)
            axes[0, 0].set_title('Cross-Validation AUC Scores')
            axes[0, 0].set_ylabel('AUC')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Confusion matrix for ensemble
            if 'ensemble' in disorder_results and 'confusion_matrix' in disorder_results['ensemble']:
                cm = np.array(disorder_results['ensemble']['confusion_matrix'])
                sns.heatmap(cm, annot=True, fmt='d', ax=axes[0, 1])
                axes[0, 1].set_title('Ensemble Confusion Matrix')
            
            # Feature importance (if available)
            try:
                importance_df = self.get_feature_importance(disorder)
                if not importance_df.empty:
                    top_features = importance_df.head(10)
                    axes[1, 0].barh(range(len(top_features)), top_features['importance'])
                    axes[1, 0].set_yticks(range(len(top_features)))
                    axes[1, 0].set_yticklabels([f'Feature {i}' for i in top_features['feature']])
                    axes[1, 0].set_title('Top 10 Feature Importance')
                    axes[1, 0].set_xlabel('Importance')
            except:
                axes[1, 0].text(0.5, 0.5, 'Feature importance not available', 
                               ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Feature Importance')
            
            # Model comparison
            axes[1, 1].bar(model_names, cv_means, color=['blue', 'green', 'red', 'orange'])
            axes[1, 1].set_title('Model Comparison')
            axes[1, 1].set_ylabel('AUC')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(output_path / f"{disorder}_results.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Results plots saved to {output_path}")

def main():
    """Main training function."""
    # Load features
    features_path = "data/features/extracted_features.csv"
    if not os.path.exists(features_path):
        logger.error(f"Features file not found: {features_path}")
        return
    
    features_df = pd.read_csv(features_path)
    logger.info(f"Loaded features: {features_df.shape}")
    
    # Initialize model trainer
    trainer = MultiDisorderBaselineModel()
    
    # Train models
    logger.info("Starting model training...")
    results = trainer.train_all_models(features_df)
    
    # Save models
    trainer.save_models()
    
    # Plot results
    trainer.plot_results(results)
    
    # Print summary
    logger.info("Training completed. Results summary:")
    for disorder, disorder_results in results.items():
        if 'ensemble' in disorder_results:
            auc = disorder_results['ensemble']['cv_auc_mean']
            std = disorder_results['ensemble']['cv_auc_std']
            logger.info(f"{disorder}: AUC = {auc:.3f} ± {std:.3f}")

if __name__ == "__main__":
    main()
