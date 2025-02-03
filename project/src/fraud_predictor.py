import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, List
import logging

class FraudPredictor:
    def __init__(self):
        """Initialize the FraudPredictor"""
        self.scaler = StandardScaler()
        self.logger = logging.getLogger(__name__)
        
    def engineer_fraud_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features specific to fraud detection
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with engineered fraud features
        """
        try:
            df = df.copy()
            
            # Calculate suspicious patterns
            df['demand_volatility'] = df['Residual_Demand'].rolling(window=24).std()
            df['efficiency_deviation'] = abs(df['System_Efficiency'] - df['System_Efficiency'].rolling(window=168).mean())
            df['re_mismatch'] = abs(df['Total_RE'] - df['RE_Utilization'] * df['Total_RE_Installed_Capacity'])
            
            # Temporal patterns
            df['hour_efficiency'] = df.groupby('Hour')['System_Efficiency'].transform('mean')
            df['efficiency_anomaly'] = abs(df['System_Efficiency'] - df['hour_efficiency'])
            
            # Loss patterns
            df['loss_ratio'] = df['Total_Loss_Factor'] / df['Dispatchable_Generation']
            df['loss_trend'] = df['loss_ratio'].rolling(window=24).mean()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error engineering fraud features: {str(e)}")
            raise
            
    def prepare_fraud_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare features for fraud detection
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            Tuple[pd.DataFrame, List[str]]: Scaled features and feature names
        """
        try:
            fraud_features = [
                'demand_volatility',
                'efficiency_deviation',
                're_mismatch',
                'efficiency_anomaly',
                'loss_ratio',
                'loss_trend'
            ]
            
            X = df[fraud_features].copy()
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=fraud_features,
                index=df.index
            )
            
            return X_scaled, fraud_features
            
        except Exception as e:
            self.logger.error(f"Error preparing fraud features: {str(e)}")
            raise
            
    def predict_fraud_probability(self, X: pd.DataFrame) -> np.ndarray:
        """
        Calculate fraud probability scores
        
        Args:
            X (pd.DataFrame): Scaled features
            
        Returns:
            np.ndarray: Fraud probability scores
        """
        try:
            # Simplified fraud probability calculation based on feature combinations
            probabilities = np.zeros(len(X))
            
            # Weight different factors
            weights = {
                'demand_volatility': 0.2,
                'efficiency_deviation': 0.25,
                're_mismatch': 0.15,
                'efficiency_anomaly': 0.2,
                'loss_ratio': 0.1,
                'loss_trend': 0.1
            }
            
            for feature, weight in weights.items():
                # Normalize feature values to [0, 1] range
                normalized = (X[feature] - X[feature].min()) / (X[feature].max() - X[feature].min())
                probabilities += normalized * weight
            
            # Ensure probabilities are between 0 and 1
            probabilities = np.clip(probabilities, 0, 1)
            
            return probabilities
            
        except Exception as e:
            self.logger.error(f"Error calculating fraud probabilities: {str(e)}")
            raise
            
    def analyze_fraud_patterns(self, df: pd.DataFrame, fraud_probs: np.ndarray) -> Dict:
        """
        Analyze patterns in fraud probabilities
        
        Args:
            df (pd.DataFrame): Input DataFrame
            fraud_probs (np.ndarray): Fraud probability scores
            
        Returns:
            Dict: Analysis results
        """
        try:
            analysis = {
                'risk_thresholds': {
                    'high': 0.7,
                    'medium': 0.4,
                    'low': 0.2
                },
                'temporal_patterns': {
                    'hourly': {},
                    'daily': {}
                },
                'risk_factors': {}
            }
            
            # Calculate temporal patterns
            df_temp = df.copy()
            df_temp['fraud_probability'] = fraud_probs
            
            # Hourly patterns
            hourly_risk = df_temp.groupby('Hour')['fraud_probability'].mean().to_dict()
            analysis['temporal_patterns']['hourly'] = hourly_risk
            
            # Daily patterns
            daily_risk = df_temp.groupby('DayOfWeek')['fraud_probability'].mean().to_dict()
            analysis['temporal_patterns']['daily'] = daily_risk
            
            # Calculate risk factor correlations
            risk_factors = [
                'demand_volatility',
                'efficiency_deviation',
                're_mismatch',
                'efficiency_anomaly',
                'loss_ratio',
                'loss_trend'
            ]
            
            for factor in risk_factors:
                correlation = np.corrcoef(df_temp[factor], fraud_probs)[0, 1]
                analysis['risk_factors'][factor] = correlation
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing fraud patterns: {str(e)}")
            raise