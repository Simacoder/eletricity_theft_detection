import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from fastapi import FastAPI
import uvicorn

# Load dataset
def load_data(filename):
    df = pd.read_csv(filename)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Feature extraction
    df['Hour'] = df['Timestamp'].dt.hour
    df['Day'] = df['Timestamp'].dt.day
    df['Month'] = df['Timestamp'].dt.month
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
    
    # Feature engineering
    df['Consumption_per_voltage'] = df['Energy Consumption (kWh)'] / df['Voltage (kV)']
    df['Power_efficiency'] = df['Power Factor'] * df['Energy Consumption (kWh)']
    
    return df

# Load dataset
df = load_data("data/smart_meter_grid_south_africa.csv")

# Feature selection
features = ['Hour', 'Day', 'Month', 'DayOfWeek', 'Energy Consumption (kWh)', 'Voltage (kV)', 'Frequency (Hz)', 'Power Factor']
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Isolation Forest for fraud detection
iso_forest = IsolationForest(contamination=0.1, random_state=42)
df['Anomaly'] = iso_forest.fit_predict(X_scaled)

# Convert anomaly results (1 = normal, -1 = fraud)
df['Fraud'] = (df['Anomaly'] == -1).astype(int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, df['Fraud'], test_size=0.2, random_state=42)

# Train Fraud Prediction Model (Random Forest)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
    
    print(f"✅ Model Evaluation Metrics:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

# Evaluate the model
evaluate_model(rf_model, X_test, y_test)

def fraud_map(df):
    # Convert fraud locations into GeoDataFrame
    fraud_data = df[df['Fraud'] == 1]
    geometry = [Point(xy) for xy in zip(fraud_data['Longitude'], fraud_data['Latitude'])]
    gdf = gpd.GeoDataFrame(fraud_data, geometry=geometry)

    # Load South Africa Map
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    sa_map = world[world.name == 'South Africa']

    # Plot fraud locations
    ax = sa_map.plot(figsize=(12, 8), color='lightgrey')
    gdf.plot(ax=ax, marker='o', color='red', markersize=5)
    plt.title('Geographic Distribution of Fraudulent Meters')
    plt.show()

# Generate fraud map
fraud_map(df)

app = FastAPI()

@app.get("/fraudsters")
def get_fraudsters():
    fraud_cases = df[df['Fraud'] == 1][['Meter ID', 'City', 'Province', 'Latitude', 'Longitude']].to_dict(orient="records")
    return {"fraudsters": fraud_cases}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
