"""
Statistical feature extraction and normalization for voice analysis.
Handles feature aggregation, normalization, and demographic adjustments.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class StatisticalFeatureProcessor:
    """Statistical processing and normalization of voice features."""
    
    def __init__(self):
        self.feature_stats = {}
        self.demographic_norms = {}
    
    def compute_feature_statistics(self, features_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Compute statistical summaries for all features.
        
        Args:
            features_df: DataFrame with extracted features
            
        Returns:
            Dictionary with feature statistics
        """
        stats_dict = {}
        
        # Get numeric feature columns (exclude metadata)
        exclude_cols = ['participant_id', 'dataset', 'age', 'gender', 
                       'depression_label', 'anxiety_label', 'ptsd_label', 'cognitive_label']
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        
        for col in feature_cols:
            if features_df[col].dtype in ['float64', 'int64']:
                values = features_df[col].dropna()
                if len(values) > 0:
                    stats_dict[col] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'median': np.median(values),
                        'q25': np.percentile(values, 25),
                        'q75': np.percentile(values, 75),
                        'min': np.min(values),
                        'max': np.max(values),
                        'skewness': stats.skew(values),
                        'kurtosis': stats.kurtosis(values)
                    }
        
        self.feature_stats = stats_dict
        return stats_dict
    
    def normalize_features(self, features_df: pd.DataFrame, 
                          method: str = 'zscore') -> pd.DataFrame:
        """
        Normalize features using specified method.
        
        Args:
            features_df: DataFrame with features
            method: Normalization method ('zscore', 'minmax', 'robust')
            
        Returns:
            DataFrame with normalized features
        """
        normalized_df = features_df.copy()
        
        # Get numeric feature columns
        exclude_cols = ['participant_id', 'dataset', 'age', 'gender', 
                       'depression_label', 'anxiety_label', 'ptsd_label', 'cognitive_label']
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        
        for col in feature_cols:
            if col in self.feature_stats:
                values = features_df[col].dropna()
                if len(values) > 0:
                    if method == 'zscore':
                        # Z-score normalization
                        mean = self.feature_stats[col]['mean']
                        std = self.feature_stats[col]['std']
                        if std > 0:
                            normalized_df[col] = (features_df[col] - mean) / std
                        else:
                            normalized_df[col] = 0
                    
                    elif method == 'minmax':
                        # Min-max normalization to [0, 1]
                        min_val = self.feature_stats[col]['min']
                        max_val = self.feature_stats[col]['max']
                        if max_val > min_val:
                            normalized_df[col] = (features_df[col] - min_val) / (max_val - min_val)
                        else:
                            normalized_df[col] = 0
                    
                    elif method == 'robust':
                        # Robust normalization using median and IQR
                        median = self.feature_stats[col]['median']
                        q75 = self.feature_stats[col]['q75']
                        q25 = self.feature_stats[col]['q25']
                        iqr = q75 - q25
                        if iqr > 0:
                            normalized_df[col] = (features_df[col] - median) / iqr
                        else:
                            normalized_df[col] = 0
        
        return normalized_df
    
    def compute_demographic_norms(self, features_df: pd.DataFrame) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Compute demographic-specific norms for features.
        
        Args:
            features_df: DataFrame with features and demographics
            
        Returns:
            Dictionary with demographic norms
        """
        norms = {}
        
        # Get numeric feature columns
        exclude_cols = ['participant_id', 'dataset', 'age', 'gender', 
                       'depression_label', 'anxiety_label', 'ptsd_label', 'cognitive_label']
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        
        # Age groups
        age_groups = {
            'young': (18, 30),
            'middle': (31, 50),
            'older': (51, 80)
        }
        
        # Gender groups
        genders = features_df['gender'].unique()
        
        for age_group, (min_age, max_age) in age_groups.items():
            norms[age_group] = {}
            
            for gender in genders:
                # Filter data for this demographic group
                mask = (features_df['age'] >= min_age) & (features_df['age'] <= max_age) & (features_df['gender'] == gender)
                group_data = features_df[mask]
                
                if len(group_data) > 0:
                    norms[age_group][gender] = {}
                    
                    for col in feature_cols:
                        if col in group_data.columns:
                            values = group_data[col].dropna()
                            if len(values) > 0:
                                norms[age_group][gender][col] = {
                                    'mean': np.mean(values),
                                    'std': np.std(values),
                                    'median': np.median(values),
                                    'count': len(values)
                                }
        
        self.demographic_norms = norms
        return norms
    
    def compute_demographic_deviations(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute deviations from demographic norms.
        
        Args:
            features_df: DataFrame with features and demographics
            
        Returns:
            DataFrame with deviation features
        """
        deviation_df = features_df.copy()
        
        # Get numeric feature columns
        exclude_cols = ['participant_id', 'dataset', 'age', 'gender', 
                       'depression_label', 'anxiety_label', 'ptsd_label', 'cognitive_label']
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        
        # Add deviation columns
        for col in feature_cols:
            deviation_col = f"{col}_deviation"
            deviation_df[deviation_col] = 0.0
            
            for idx, row in features_df.iterrows():
                age = row['age']
                gender = row['gender']
                
                # Determine age group
                if 18 <= age <= 30:
                    age_group = 'young'
                elif 31 <= age <= 50:
                    age_group = 'middle'
                elif 51 <= age <= 80:
                    age_group = 'older'
                else:
                    continue
                
                # Get demographic norm
                if (age_group in self.demographic_norms and 
                    gender in self.demographic_norms[age_group] and
                    col in self.demographic_norms[age_group][gender]):
                    
                    norm_mean = self.demographic_norms[age_group][gender][col]['mean']
                    norm_std = self.demographic_norms[age_group][gender][col]['std']
                    
                    if norm_std > 0:
                        deviation = (row[col] - norm_mean) / norm_std
                        deviation_df.loc[idx, deviation_col] = deviation
        
        return deviation_df
    
    def create_feature_groups(self, features_df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Group features by type for analysis.
        
        Args:
            features_df: DataFrame with features
            
        Returns:
            Dictionary with feature groups
        """
        feature_groups = {
            'egemaps': [],
            'prosodic': [],
            'voice_quality': [],
            'energy': [],
            'spectral': [],
            'formant': [],
            'timing': [],
            'rhythm': []
        }
        
        for col in features_df.columns:
            if col.startswith('F0') or col.startswith('f0') or 'pitch' in col.lower():
                feature_groups['prosodic'].append(col)
            elif 'jitter' in col.lower() or 'shimmer' in col.lower() or 'hnr' in col.lower():
                feature_groups['voice_quality'].append(col)
            elif 'rms' in col.lower() or 'energy' in col.lower() or 'loudness' in col.lower():
                feature_groups['energy'].append(col)
            elif 'spectral' in col.lower() or 'mfcc' in col.lower() or 'zcr' in col.lower():
                feature_groups['spectral'].append(col)
            elif col.startswith('F') and col[1].isdigit():  # F1, F2, F3, etc.
                feature_groups['formant'].append(col)
            elif 'pause' in col.lower() or 'speech_rate' in col.lower() or 'timing' in col.lower():
                feature_groups['timing'].append(col)
            elif 'rhythm' in col.lower() or 'tempo' in col.lower() or 'onset' in col.lower():
                feature_groups['rhythm'].append(col)
            else:
                # Check if it's an eGeMAPS feature (typically has specific naming)
                if any(keyword in col.lower() for keyword in ['mfcc', 'spectral', 'energy', 'voicing']):
                    feature_groups['egemaps'].append(col)
        
        return feature_groups
    
    def compute_feature_correlations(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute feature correlations.
        
        Args:
            features_df: DataFrame with features
            
        Returns:
            Correlation matrix
        """
        # Get numeric feature columns
        exclude_cols = ['participant_id', 'dataset', 'age', 'gender', 
                       'depression_label', 'anxiety_label', 'ptsd_label', 'cognitive_label']
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        
        # Compute correlation matrix
        correlation_matrix = features_df[feature_cols].corr()
        
        return correlation_matrix
    
    def select_uncorrelated_features(self, features_df: pd.DataFrame, 
                                   threshold: float = 0.95) -> List[str]:
        """
        Select features with low correlation to avoid multicollinearity.
        
        Args:
            features_df: DataFrame with features
            threshold: Correlation threshold for feature removal
            
        Returns:
            List of selected feature names
        """
        # Get numeric feature columns
        exclude_cols = ['participant_id', 'dataset', 'age', 'gender', 
                       'depression_label', 'anxiety_label', 'ptsd_label', 'cognitive_label']
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        
        # Compute correlation matrix
        corr_matrix = features_df[feature_cols].corr().abs()
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > threshold:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
        
        # Remove one feature from each highly correlated pair
        features_to_remove = set()
        for feat1, feat2 in high_corr_pairs:
            # Keep the feature with higher variance
            var1 = features_df[feat1].var()
            var2 = features_df[feat2].var()
            
            if var1 < var2:
                features_to_remove.add(feat1)
            else:
                features_to_remove.add(feat2)
        
        # Return selected features
        selected_features = [col for col in feature_cols if col not in features_to_remove]
        
        logger.info(f"Selected {len(selected_features)} features from {len(feature_cols)} total features")
        logger.info(f"Removed {len(features_to_remove)} highly correlated features")
        
        return selected_features
    
    def create_feature_summary(self, features_df: pd.DataFrame) -> Dict:
        """
        Create comprehensive feature summary.
        
        Args:
            features_df: DataFrame with features
            
        Returns:
            Dictionary with feature summary
        """
        summary = {
            'total_participants': len(features_df),
            'total_features': len([col for col in features_df.columns 
                                 if col not in ['participant_id', 'dataset', 'age', 'gender', 
                                               'depression_label', 'anxiety_label', 'ptsd_label', 'cognitive_label']]),
            'datasets': features_df['dataset'].value_counts().to_dict(),
            'age_distribution': {
                'mean': features_df['age'].mean(),
                'std': features_df['age'].std(),
                'min': features_df['age'].min(),
                'max': features_df['age'].max()
            },
            'gender_distribution': features_df['gender'].value_counts().to_dict(),
            'label_distributions': {
                'depression': features_df['depression_label'].value_counts().to_dict(),
                'anxiety': features_df['anxiety_label'].value_counts().to_dict(),
                'ptsd': features_df['ptsd_label'].value_counts().to_dict(),
                'cognitive': features_df['cognitive_label'].value_counts().to_dict()
            }
        }
        
        return summary
