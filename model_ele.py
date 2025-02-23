import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')
import plotly.express as px
from sklearn.decomposition import PCA

# 1. Data Loading and Preprocessing
async def load_and_preprocess():
    # Read the CSV file
    file_content = await window.fs.readFile('data/smart_meter_grid_south_africa.csv', {'encoding': 'utf8'})
    df = pd.read_csv(pd.StringIO(file_content))
    
    # Convert timestamp to datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Extract time features
    df['Hour'] = df['Timestamp'].dt.hour
    df['Day'] = df['Timestamp'].dt.day
    df['Month'] = df['Timestamp'].dt.month
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
    
    # Create relevant features for anomaly detection
    df['Consumption_per_voltage'] = df['Energy Consumption (kWh)'] / df['Voltage (kV)']
    df['Power_efficiency'] = df['Power Factor'] * df['Energy Consumption (kWh)']
    
    return df

# 2. Feature Selection and Scaling
def prepare_features(df):
    # Select features for anomaly detection
    features = [
        'Energy Consumption (kWh)',
        'Voltage (kV)',
        'Frequency (Hz)',
        'Power Factor',
        'Consumption_per_voltage',
        'Power_efficiency'
    ]
    
    X = df[features]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, features

# 3. Implement Multiple Anomaly Detection Methods
def detect_anomalies(X_scaled):
    # Isolation Forest
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    iso_forest_labels = iso_forest.fit_predict(X_scaled)
    
    # Local Outlier Factor
    lof = LocalOutlierFactor(contamination=0.1, n_neighbors=20)
    lof_labels = lof.fit_predict(X_scaled)
    
    # Robust Covariance (Elliptic Envelope)
    robust_cov = EllipticEnvelope(contamination=0.1, random_state=42)
    robust_cov_labels = robust_cov.fit_predict(X_scaled)
    
    # One-Class SVM
    one_class_svm = OneClassSVM(kernel='rbf', nu=0.1)
    one_class_svm_labels = one_class_svm.fit_predict(X_scaled)
    
    return {
        'Isolation Forest': iso_forest_labels,
        'Local Outlier Factor': lof_labels,
        'Robust Covariance': robust_cov_labels,
        'One-Class SVM': one_class_svm_labels
    }

# 4. Visualize Results
def visualize_anomalies(df, X_scaled, features, anomaly_labels):
    # PCA for dimensionality reduction
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create visualization for each method
    for method, labels in anomaly_labels.items():
        plt.figure(figsize=(10, 6))
        plt.scatter(X_pca[labels == 1, 0], X_pca[labels == 1, 1], 
                   c='blue', label='Normal')
        plt.scatter(X_pca[labels == -1, 0], X_pca[labels == -1, 1], 
                   c='red', label='Anomaly')
        plt.title(f'Anomaly Detection using {method}')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.legend()
        plt.show()
    
    # Visualize energy consumption patterns
    plt.figure(figsize=(12, 6))
    plt.scatter(df['Hour'], df['Energy Consumption (kWh)'], 
               c=anomaly_labels['Isolation Forest'], cmap='viridis')
    plt.title('Energy Consumption Pattern with Anomalies')
    plt.xlabel('Hour of Day')
    plt.ylabel('Energy Consumption (kWh)')
    plt.colorbar(label='Anomaly Score')
    plt.show()

# 5. Compare Results with Known Fraud Cases
def compare_with_fraud(df, anomaly_labels):
    for method, labels in anomaly_labels.items():
        # Convert labels to binary (1 for normal, 0 for anomaly)
        predicted_anomalies = (labels == -1).astype(int)
        actual_fraud = df['Fraud'].values
        
        print(f"\nResults for {method}:")
        print("\nClassification Report:")
        print(classification_report(actual_fraud, predicted_anomalies))
        
        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix(actual_fraud, predicted_anomalies), 
                   annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {method}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

# 6. Time Series Analysis for Anomaly Detection
def analyze_time_patterns(df, anomaly_labels):
    # Group by hour and calculate anomaly percentages
    hourly_anomalies = pd.DataFrame()
    for method, labels in anomaly_labels.items():
        anomaly_series = (labels == -1).astype(int)
        hourly_stats = pd.DataFrame({
            'Hour': df['Hour'],
            'Anomaly': anomaly_series
        }).groupby('Hour')['Anomaly'].mean()
        hourly_anomalies[method] = hourly_stats
    
    # Visualize hourly patterns
    plt.figure(figsize=(12, 6))
    hourly_anomalies.plot()
    plt.title('Hourly Anomaly Patterns')
    plt.xlabel('Hour of Day')
    plt.ylabel('Percentage of Anomalies')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# 7. Predict Future Anomalies
def build_prediction_model(df, anomaly_labels):
    # Prepare features for prediction
    features = [
        'Hour', 'Day', 'Month', 'DayOfWeek',
        'Energy Consumption (kWh)', 'Voltage (kV)',
        'Frequency (Hz)', 'Power Factor'
    ]
    X = df[features]
    
    # Use Isolation Forest anomalies as target
    y = (anomaly_labels['Isolation Forest'] == -1).astype(int)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a Random Forest classifier
    from sklearn.ensemble import RandomForestClassifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)
    
    # Evaluate predictions
    y_pred = rf_classifier.predict(X_test)
    print("\nAnomaly Prediction Model Results:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': rf_classifier.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Feature Importance for Anomaly Prediction')
    plt.show()
    
    return rf_classifier

# Main execution
async def main():
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = await load_and_preprocess()
    
    # Prepare features
    print("Preparing features...")
    X_scaled, features = prepare_features(df)
    
    # Detect anomalies
    print("Detecting anomalies...")
    anomaly_labels = detect_anomalies(X_scaled)
    
    # Visualize results
    print("Visualizing results...")
    visualize_anomalies(df, X_scaled, features, anomaly_labels)
    
    # Compare with known fraud cases
    print("Comparing with known fraud cases...")
    compare_with_fraud(df, anomaly_labels)
    
    # Analyze time patterns
    print("Analyzing time patterns...")
    analyze_time_patterns(df, anomaly_labels)
    
    # Build prediction model
    print("Building prediction model...")
    prediction_model = build_prediction_model(df, anomaly_labels)
    
    return df, anomaly_labels, prediction_model

# Run the analysis
    df, anomaly_labels, prediction_model = await main()