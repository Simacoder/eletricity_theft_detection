import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

class PowerSystemAnalytics:
    def process_data(self, data_path):
        try:
            df = pd.read_csv(data_path, encoding='utf-8', encoding_errors='ignore')
            df = self.prepare_datetime(df)
            results = self.detect_anomalies(df)
            stats = self.calculate_statistics(results)
            return df, results, stats
        except Exception as e:
            st.error(f"Failed to process data: {str(e)}")
            return None, None, None

    def prepare_datetime(self, df):
        datetime_columns = [col for col in df.columns if any(term in col.lower() 
                          for term in ['date', 'time', 'timestamp'])]
        
        if 'Date Time Hour Beginning' not in df.columns and datetime_columns:
            df['Date Time Hour Beginning'] = pd.to_datetime(df[datetime_columns[0]], 
                                                          errors='coerce')
        elif 'Date Time Hour Beginning' in df.columns:
            df['Date Time Hour Beginning'] = pd.to_datetime(df['Date Time Hour Beginning'], 
                                                          errors='coerce')
        else:
            raise ValueError("No datetime column found in the dataset")
        
        df = df.dropna(subset=['Date Time Hour Beginning'])
        df['hour'] = df['Date Time Hour Beginning'].dt.hour
        df['day'] = df['Date Time Hour Beginning'].dt.day_name()
        df['month'] = df['Date Time Hour Beginning'].dt.month
        return df

    def detect_anomalies(self, df):
        features = ['Residual Demand', 'Dispatchable Generation', 'Total RE']
        df_filtered = df[features].dropna()
        
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df_filtered)
        
        model = IsolationForest(contamination=0.05, random_state=42)
        model.fit(df_scaled)
        
        df.loc[df_filtered.index, 'anomaly_score'] = model.decision_function(df_scaled)
        df['anomaly'] = (df['anomaly_score'] < -0.1).astype(int)
        
        df['demand_ma'] = df['Residual Demand'].rolling(window=24).mean()
        df['generation_ma'] = df['Dispatchable Generation'].rolling(window=24).mean()
        
        return df

    def calculate_statistics(self, df):
        stats = {
            'total_anomalies': int(df['anomaly'].sum()),
            'anomaly_rate': float(df['anomaly'].mean() * 100),
            'hourly_patterns': df[df['anomaly'] == 1]['hour'].value_counts().to_dict(),
            'daily_patterns': df[df['anomaly'] == 1]['day'].value_counts().to_dict(),
            'monthly_patterns': df[df['anomaly'] == 1]['month'].value_counts().to_dict(),
            'demand_stats': {
                'mean': float(df['Residual Demand'].mean()),
                'std': float(df['Residual Demand'].std()),
                'min': float(df['Residual Demand'].min()),
                'max': float(df['Residual Demand'].max())
            }
        }
        return stats

    def create_visualizations(self, df):
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Power Demand and Generation', 
                          'Anomaly Detection', 
                          'Renewable Energy Integration'),
            vertical_spacing=0.1,
            specs=[[{"secondary_y": True}],
                  [{"secondary_y": False}],
                  [{"secondary_y": False}]]
        )

        fig.add_trace(
            go.Scatter(x=df['Date Time Hour Beginning'], 
                      y=df['Residual Demand'],
                      name='Demand',
                      line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['Date Time Hour Beginning'], 
                      y=df['demand_ma'],
                      name='Demand (24h MA)',
                      line=dict(color='blue', dash='dash')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['Date Time Hour Beginning'], 
                      y=df['Dispatchable Generation'],
                      name='Generation',
                      line=dict(color='green')),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=df['Date Time Hour Beginning'], 
                      y=df['anomaly_score'],
                      name='Anomaly Score',
                      line=dict(color='red')),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(x=df['Date Time Hour Beginning'], 
                      y=df['Total RE'],
                      name='Renewable Energy',
                      fill='tozeroy',
                      line=dict(color='orange')),
            row=3, col=1
        )

        fig.update_layout(
            height=1200,
            showlegend=True,
            title_text="Power System Analytics Dashboard"
        )
        
        st.plotly_chart(fig, use_container_width=True)

def main():
    st.set_page_config(layout="wide")
    st.title("Power System Analytics & Anomaly Detection")
    
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file:
        psa = PowerSystemAnalytics()
        df, results, stats = psa.process_data(uploaded_file)
        
        if df is not None:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Anomalies", stats['total_anomalies'])
            with col2:
                st.metric("Anomaly Rate", f"{stats['anomaly_rate']:.2f}%")
            with col3:
                st.metric("Average Demand", f"{stats['demand_stats']['mean']:.2f} MW")
            
            st.subheader("Processed Data Sample")
            st.dataframe(df.head())
            
            st.subheader("Anomaly Patterns")
            col1, col2 = st.columns(2)
            with col1:
                st.write("Hourly Patterns", stats['hourly_patterns'])
            with col2:
                st.write("Daily Patterns", stats['daily_patterns'])
            
            psa.create_visualizations(results)
        else:
            st.error("Failed to process the uploaded data.")

if __name__ == "__main__":
    main()