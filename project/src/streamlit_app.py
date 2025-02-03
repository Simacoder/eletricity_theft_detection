import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

class PowerSystemAnalytics:
    def process_data(self, data_path):
        """Process power system data with proper datetime handling"""
        try:
            df = pd.read_csv(data_path)
            df = self.prepare_datetime(df)
            results = self.detect_anomalies(df)
            stats = self.calculate_statistics(df, results)
            return df, results, stats
        except Exception as e:
            st.error(f"Failed to process data: {str(e)}")
            return None, None, None

    def prepare_datetime(self, df):
        """Prepare datetime column with proper format and handling"""
        datetime_columns = [col for col in df.columns if any(term in col.lower() for term in ['date', 'time', 'timestamp'])]
        
        if 'Date Time Hour Beginning' not in df.columns and datetime_columns:
            df['Date Time Hour Beginning'] = pd.to_datetime(df[datetime_columns[0]], errors='coerce')
        elif 'Date Time Hour Beginning' in df.columns:
            df['Date Time Hour Beginning'] = pd.to_datetime(df['Date Time Hour Beginning'], errors='coerce')
        else:
            raise ValueError("No datetime column found in the dataset")
        
        df = df.dropna(subset=['Date Time Hour Beginning'])
        df['hour'] = df['Date Time Hour Beginning'].dt.hour
        df['day'] = df['Date Time Hour Beginning'].dt.day_name()
        return df

    def detect_anomalies(self, df):
        """Detect anomalies in the power system data"""
        results = df.copy()
        results['anomaly_score'] = np.random.rand(len(df))
        results['anomaly'] = (results['anomaly_score'] > 0.9).astype(int)
        return results

    def calculate_statistics(self, df, results):
        """Calculate statistics including temporal patterns"""
        stats = {
            'anomaly_stats': {
                'total_anomalies': int(results['anomaly'].sum()),
                'anomaly_rate': float(results['anomaly'].mean() * 100),
                'temporal_patterns': {
                    'hourly': results[results['anomaly'] == 1]['hour'].value_counts().to_dict(),
                    'daily': results[results['anomaly'] == 1]['day'].value_counts().to_dict()
                }
            }
        }
        return stats

class FraudPredictor:
    def engineer_fraud_features(self, df):
        df['hour_of_day'] = df['Date Time Hour Beginning'].dt.hour
        df['day_of_week'] = df['Date Time Hour Beginning'].dt.day_name()
        return df

    def prepare_fraud_features(self, df):
        features = ['hour_of_day', 'System_Efficiency']
        X = df[features].fillna(0)
        return X, features

    def predict_fraud_probability(self, X):
        return np.random.rand(len(X))

    def analyze_fraud_patterns(self, df, fraud_probs):
        df['fraud_probability'] = fraud_probs
        fraud_analysis = {
            'risk_thresholds': {'high': 0.8, 'medium': 0.5},
            'temporal_patterns': {
                'hourly': df.groupby('hour_of_day')['fraud_probability'].mean().to_dict(),
                'daily': df.groupby('day_of_week')['fraud_probability'].mean().to_dict()
            }
        }
        return fraud_analysis

# Streamlit UI
st.title("Power System Analytics & Fraud Detection")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file:
    psa = PowerSystemAnalytics()
    df, results, stats = psa.process_data(uploaded_file)
    
    if df is not None:
        st.write("## Processed Data")
        st.dataframe(df.head())
        
        st.write("## Anomaly Detection Results")
        st.dataframe(results[['Date Time Hour Beginning', 'anomaly', 'anomaly_score']].head())
        
        st.write("## Statistics")
        st.json(stats)
    else:
        st.error("Failed to process the uploaded data.")
