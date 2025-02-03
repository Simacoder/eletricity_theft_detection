import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import logging
from typing import Dict, Union, Tuple
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data_processing.log')
    ]
)

logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, use_robust_scaler: bool = True):
        self.scaler = RobustScaler() if use_robust_scaler else StandardScaler()
        self.normalizer = MinMaxScaler()
        self.feature_columns = None
        self.logger = logging.getLogger(__name__)
        
    def load_data(self, file_path: Union[str, Path]) -> Tuple[pd.DataFrame, pd.DatetimeIndex]:
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            df = pd.read_csv(file_path)
            self.logger.info(f"Successfully loaded data from {file_path}. Shape: {df.shape}")
            
            self.logger.info(f"Columns in dataset: {df.columns.tolist()}")
            
            datetime_index = None
            if 'Date Time Hour Beginning' in df.columns:
                datetime_index = pd.to_datetime(df['Date Time Hour Beginning'], errors='coerce')
                df['hour'] = datetime_index.dt.hour  # Extract hour
                df['day'] = datetime_index.dt.day_name()  # Extract day
                df = df.drop('Date Time Hour Beginning', axis=1)
                self.logger.info("Successfully extracted datetime index.")
            else:
                self.logger.warning("No recognizable datetime column found. Proceeding without indexing by date.")
            
            return df, datetime_index
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            self.logger.info("Starting data cleaning process...")
            initial_shape = df.shape
            df = df.drop_duplicates()
            df = df.replace([np.inf, -np.inf], np.nan)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            categorical_cols = df.select_dtypes(exclude=[np.number]).columns
            if not categorical_cols.empty:
                df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
            df[numeric_cols] = self.normalizer.fit_transform(df[numeric_cols])
            self.logger.info(f"Data cleaning completed. Shape: {df.shape}")
            return df
        except Exception as e:
            self.logger.error(f"Error in data cleaning: {str(e)}")
            raise

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            self.logger.info("Starting feature engineering...")
            rename_mapping = {
                'Dispatchable Generation': 'Dispatchable_Generation',
                'Residual Demand': 'Demand',
                'RSA Contracted Forecast': 'Supply',
                'Residual Forecast': 'Residual_Forecast'
            }
            df = df.rename(columns=rename_mapping)
            required_columns = ['Demand', 'Supply', 'Dispatchable_Generation']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            df['Generation_Demand_Ratio'] = df['Dispatchable_Generation'] / df['Demand'].replace(0, np.nan)
            df['RE_Utilization'] = df['Supply'] / df['Demand'].replace(0, np.nan)
            df['Supply_Demand_Gap'] = df['Supply'] - df['Demand']
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.fillna(df.mean())
            self.logger.info("Feature engineering completed successfully")
            return df
        except Exception as e:
            self.logger.error(f"Error in feature engineering: {str(e)}")
            raise

    def prepare_features(self, df: pd.DataFrame, datetime_index: pd.DatetimeIndex = None) -> pd.DataFrame:
        try:
            self.logger.info("Preparing features for modeling...")
            features = ['Demand', 'Supply', 'Generation_Demand_Ratio', 'RE_Utilization', 'Supply_Demand_Gap']
            missing_features = [f for f in features if f not in df.columns]
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
            if datetime_index is not None:
                df['hour'] = datetime_index.hour
                df['day_of_week'] = datetime_index.day_name()
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(df[features]), 
                columns=features, 
                index=datetime_index if datetime_index is not None else df.index
            )
            self.logger.info(f"Feature preparation completed. Shape: {X_scaled.shape}")
            return X_scaled
        except Exception as e:
            self.logger.error(f"Error in feature preparation: {str(e)}")
            raise

if __name__ == "__main__":
    try:
        preprocessor = DataPreprocessor()
        data, datetime_index = preprocessor.load_data("data/ESK10705.csv")
        cleaned_data = preprocessor.clean_data(data)
        engineered_data = preprocessor.engineer_features(cleaned_data)
        processed_features = preprocessor.prepare_features(engineered_data, datetime_index)
        print(f"Final processed data shape: {processed_features.shape}")
    except Exception as e:
        logger.error(f"Failed to process data: {str(e)}")
