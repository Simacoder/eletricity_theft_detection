import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import logging
from typing import Dict, Union, Optional, Tuple
from pathlib import Path
import joblib
from preprocessing import DataPreprocessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('modeling.log')
    ]
)

logger = logging.getLogger(__name__)

class EnhancedAnomalyDetector:
    def __init__(
        self,
        contamination: float = 0.1,
        n_estimators: int = 200,
        max_features: float = 1.0,
        random_state: int = 42
    ):
        """
        Initialize the anomaly detector
        
        Args:
            contamination (float): Expected proportion of outliers in the data
            n_estimators (int): Number of base estimators
            max_features (float): Number of features to draw from X
            random_state (int): Random state for reproducibility
        """
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_features=max_features,
            random_state=random_state,
            n_jobs=-1
        )
        self.metrics = {}
        self.logger = logging.getLogger(__name__)

    def train(self, X: pd.DataFrame, validation_split: float = 0.2) -> Dict:
        """
        Train the model and compute metrics
        
        Args:
            X (pd.DataFrame): Feature matrix
            validation_split (float): Validation set size
            
        Returns:
            Dict: Training metrics
        """
        try:
            self.logger.info("Training anomaly detection model...")
            
            # Split data
            X_train, X_val = train_test_split(
                X,
                test_size=validation_split,
                random_state=42
            )
            
            # Train model
            self.model.fit(X_train)
            
            # Get predictions and scores
            train_pred = self.model.predict(X_train)
            train_scores = self.model.score_samples(X_train)
            val_pred = self.model.predict(X_val)
            val_scores = self.model.score_samples(X_val)
            
            # Calculate metrics
            self.metrics = {
                'train': {
                    'anomaly_ratio': (train_pred == -1).mean(),
                    'score_stats': {
                        'mean': float(np.mean(train_scores)),
                        'std': float(np.std(train_scores)),
                        'percentiles': {
                            '1%': float(np.percentile(train_scores, 1)),
                            '5%': float(np.percentile(train_scores, 5)),
                            '95%': float(np.percentile(train_scores, 95)),
                            '99%': float(np.percentile(train_scores, 99))
                        }
                    }
                },
                'validation': {
                    'anomaly_ratio': (val_pred == -1).mean(),
                    'score_stats': {
                        'mean': float(np.mean(val_scores)),
                        'std': float(np.std(val_scores)),
                        'percentiles': {
                            '1%': float(np.percentile(val_scores, 1)),
                            '5%': float(np.percentile(val_scores, 5)),
                            '95%': float(np.percentile(val_scores, 95)),
                            '99%': float(np.percentile(val_scores, 99))
                        }
                    }
                }
            }
            
            self.logger.info("Model training completed successfully")
            return self.metrics
            
        except Exception as e:
            self.logger.error(f"Error in model training: {str(e)}")
            raise

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new data
        
        Args:
            X (pd.DataFrame): Feature matrix
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Predictions (-1 for anomaly, 1 for normal) and anomaly scores
        """
        try:
            predictions = self.model.predict(X)
            scores = self.model.score_samples(X)
            self.logger.info(f"Generated predictions for {len(predictions)} samples")
            return predictions, scores
            
        except Exception as e:
            self.logger.error(f"Error in prediction: {str(e)}")
            raise

    def analyze_anomalies(self, df: pd.DataFrame, predictions: np.ndarray, scores: np.ndarray) -> Tuple[pd.DataFrame, Dict]:
        """
        Analyze anomalies and generate statistics
        
        Args:
            df (pd.DataFrame): Original DataFrame
            predictions (np.ndarray): Model predictions
            scores (np.ndarray): Anomaly scores
            
        Returns:
            Tuple[pd.DataFrame, Dict]: Results DataFrame and anomaly statistics
        """
        try:
            # Create results DataFrame
            results = df.copy()
            results['anomaly'] = predictions
            results['anomaly_score'] = scores
            
            # Calculate temporal patterns
            results['hour'] = pd.to_datetime(results['Date Time Hour Beginning']).dt.hour
            results['day'] = pd.to_datetime(results['Date Time Hour Beginning']).dt.day_name()
            
            # Generate statistics
            anomaly_stats = {
                'overall': {
                    'total_anomalies': int((predictions == -1).sum()),
                    'anomaly_rate': float((predictions == -1).mean() * 100)
                },
                'temporal_patterns': {
                    'hourly': results[results['anomaly'] == -1]['hour'].value_counts().to_dict(),
                    'daily': results[results['anomaly'] == -1]['day'].value_counts().to_dict()
                }
            }
            
            return results, anomaly_stats
            
        except Exception as e:
            self.logger.error(f"Error in anomaly analysis: {str(e)}")
            raise

    def save_model(self, filepath: Union[str, Path]) -> None:
        """
        Save the trained model
        
        Args:
            filepath (Union[str, Path]): Path to save the model
        """
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.model, filepath)
            self.logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self, filepath: Union[str, Path]) -> None:
        """
        Load a saved model
        
        Args:
            filepath (Union[str, Path]): Path to the saved model
        """
        try:
            filepath = Path(filepath)
            if not filepath.exists():
                raise FileNotFoundError(f"Model file not found: {filepath}")
                
            self.model = joblib.load(filepath)
            self.logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

def process_pipeline(
    file_path: Union[str, Path],
    model_save_path: Optional[Union[str, Path]] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Run the complete anomaly detection pipeline
    
    Args:
        file_path (Union[str, Path]): Path to input data
        model_save_path (Optional[Union[str, Path]]): Path to save trained model
        
    Returns:
        Tuple[pd.DataFrame, Dict]: Processed data with predictions and metrics
    """
    try:
        logger.info("Starting anomaly detection pipeline...")
        
        # Initialize components
        preprocessor = DataPreprocessor(use_robust_scaler=True)
        detector = EnhancedAnomalyDetector(contamination=0.1)
        
        # Process data
        raw_data = preprocessor.load_data(file_path)
        cleaned_data = preprocessor.clean_data(raw_data)
        engineered_data = preprocessor.engineer_features(cleaned_data)
        processed_features = preprocessor.prepare_features(engineered_data)
        
        # Train model and get predictions
        metrics = detector.train(processed_features)
        predictions = detector.predict(processed_features)
        
        # Add predictions to features
        processed_features['anomaly'] = predictions
        
        # Save model if path provided
        if model_save_path:
            detector.save_model(model_save_path)
        
        logger.info("Anomaly detection pipeline completed successfully")
        return processed_features, metrics
        
    except Exception as e:
        logger.error(f"Error in anomaly detection pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Example usage
        file_path = "data/ESK10705.csv"
        model_save_path = "models/anomaly_detector.joblib"
        processed_data, metrics = process_pipeline(file_path, model_save_path)
        print("Pipeline completed successfully")
        print(f"Processed data shape: {processed_data.shape}")
        print("Training metrics:", metrics)
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")