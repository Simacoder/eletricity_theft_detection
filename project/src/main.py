from modeling import EnhancedAnomalyDetector
from preprocessing import DataPreprocessor
import pandas as pd
import numpy as np
from typing import Tuple, Dict
import logging


class PowerSystemAnalytics:
    def __init__(self):
        """Initialize the pipeline components"""
        self.preprocessor = DataPreprocessor()
        self.detector = EnhancedAnomalyDetector()
        self.logger = logging.getLogger(__name__)

    def process_data(self, data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Process the data through the entire pipeline
        
        Args:
            data_path (str): Path to the input CSV file
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, Dict]: Original DataFrame, results DataFrame, and statistics
        """
        try:
            # Load and preprocess data - now handling datetime index separately
            df, datetime_index = self.preprocessor.load_data(data_path)
            df = self.preprocessor.clean_data(df)
            df = self.preprocessor.engineer_features(df)
            X_scaled = self.preprocessor.prepare_features(df, datetime_index)

            # Train model and get predictions
            self.detector.train(X_scaled)
            predictions, scores = self.detector.predict(X_scaled)

            # Create results DataFrame with datetime index
            results = pd.DataFrame({
                'anomaly': predictions,
                'anomaly_score': scores
            }, index=datetime_index if datetime_index is not None else X_scaled.index)

            # Get analysis results
            anomaly_stats = self._analyze_anomalies(results, datetime_index)

            # Calculate additional system metrics
            system_stats = self._calculate_system_metrics(df)

            # Calculate feature impact
            feature_impact = self._calculate_feature_impact(X_scaled, predictions)

            # Combine all stats
            combined_stats = {
                'anomaly_stats': {
                    'total_anomalies': anomaly_stats['overall']['total_anomalies'],
                    'anomaly_rate': anomaly_stats['overall']['anomaly_rate'],
                    'temporal_patterns': anomaly_stats['temporal_patterns'],
                    'feature_impact': feature_impact
                },
                'system_summary': {
                    'system_metrics': system_stats
                }
            }

            # Add datetime index back to original df if it exists
            if datetime_index is not None:
                df['Date Time Hour Beginning'] = datetime_index

            return df, results, combined_stats

        except Exception as e:
            self.logger.error(f"Error processing data: {str(e)}")
            raise

    def _analyze_anomalies(self, results: pd.DataFrame, datetime_index: pd.DatetimeIndex = None) -> Dict:
        """Analyze anomaly patterns with respect to time"""
        anomalies = results[results['anomaly'] == -1]
        
        stats = {
            'overall': {
                'total_anomalies': len(anomalies),
                'anomaly_rate': (len(anomalies) / len(results)) * 100
            },
            'temporal_patterns': {
                'hourly': {},
                'daily': {}
            }
        }

        if datetime_index is not None:
            # Hourly patterns
            hourly_counts = anomalies.groupby(datetime_index.hour).size()
            stats['temporal_patterns']['hourly'] = hourly_counts.to_dict()

            # Daily patterns
            daily_counts = anomalies.groupby(datetime_index.dayofweek).size()
            stats['temporal_patterns']['daily'] = daily_counts.to_dict()

        return stats

    def _calculate_system_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate system-wide performance metrics"""
        metrics = {}
        try:
            metrics['avg_system_efficiency'] = df['System_Efficiency'].mean()
            metrics['avg_re_contribution'] = (df['Total_RE'] / df['Dispatchable_Generation']).mean()
            metrics['avg_load_factor'] = df['Load_Factor'].mean() if 'Load_Factor' in df.columns else np.nan
            metrics['avg_loss_factor'] = df['Total_Loss_Factor'].mean() if 'Total_Loss_Factor' in df.columns else np.nan
        except Exception as e:
            self.logger.warning(f"Some metrics could not be calculated: {str(e)}")
        return metrics

    def _calculate_feature_impact(self, X: pd.DataFrame, predictions: np.ndarray) -> Dict:
        """Calculate feature impact on anomaly detection"""
        feature_impact = {}
        for feature in X.columns:
            correlation = np.abs(np.corrcoef(X[feature], predictions == -1)[0, 1])
            feature_impact[feature] = correlation
        return feature_impact


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Define data path
    data_path = "data/ESK10705.csv"
    
    # Initialize analytics pipeline
    analytics = PowerSystemAnalytics()

    # Process data
    try:
        df, results, combined_stats = analytics.process_data(data_path)

        # Display processed results
        print("\n--- Processed Data Sample ---")
        print(df.head())

        print("\n--- Anomaly Detection Results ---")
        print(results.head())

        print("\n--- Summary Statistics ---")
        print(combined_stats)

    except Exception as e:
        logging.error(f"Failed to process data: {e}")