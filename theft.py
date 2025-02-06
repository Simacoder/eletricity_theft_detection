import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv('synthetic_household_energy_data_with_metadata.csv', parse_dates=['date'])

data = load_data()

# Sidebar
st.sidebar.title("Energy Consumption Analysis")
option = st.sidebar.selectbox("Select Analysis", [
    "Dataset Overview", "Cluster Analysis", "Anomaly Detection", "Theft Prediction", "Daily Consumption Trends"])

# Dataset Overview
if option == "Dataset Overview":
    st.title("Dataset Overview")
    st.write("Preview of the dataset:")
    st.write(data.head())
    
    st.write("Basic Statistics:")
    st.write(data.describe())

# Cluster Analysis
elif option == "Cluster Analysis":
    st.title("Household Consumption Clustering")
    
    # Aggregate monthly data
    monthly_data = data.groupby(['household_id', pd.Grouper(key='date', freq='M')]).mean()
    scaled_data = monthly_data.drop(columns=['avg_temperature'], axis=1)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(scaled_data)
    
    # Determine optimal clusters (Elbow method)
    inertia = []
    K = range(1, 10)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(scaled_data)
        inertia.append(kmeans.inertia_)
    
    fig, ax = plt.subplots()
    ax.plot(K, inertia, 'bx-')
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('Inertia')
    ax.set_title('Elbow Method for Optimal k')
    st.pyplot(fig)
    
    st.write("Clustering applied with k=4")
    kmeans = KMeans(n_clusters=4, random_state=42)
    monthly_data['cluster'] = kmeans.fit_predict(scaled_data)
    
    st.write("Cluster Distribution:")
    st.write(monthly_data['cluster'].value_counts())

# Anomaly Detection
elif option == "Anomaly Detection":
    st.title("Anomaly Detection with Isolation Forest")
    
    # Train Isolation Forest
    iso_forest = IsolationForest(contamination=0.01, random_state=42)
    monthly_data = data.groupby(['household_id', pd.Grouper(key='date', freq='M')]).mean()
    scaled_data = StandardScaler().fit_transform(monthly_data.drop(columns=['avg_temperature'], axis=1))
    monthly_data['anomaly'] = iso_forest.fit_predict(scaled_data)
    monthly_data['anomaly'] = monthly_data['anomaly'].map({1: 0, -1: 1})
    
    st.write("Number of detected anomalies:", monthly_data['anomaly'].sum())
    
    # Plot anomalies over time
    anomaly_trend = monthly_data.groupby(monthly_data.index.get_level_values('date'))['anomaly'].sum()
    fig, ax = plt.subplots()
    anomaly_trend.plot(ax=ax, color='red', marker='o')
    ax.set_title("Anomalies Detected Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Anomalies")
    st.pyplot(fig)
    
    # Plot anomalies distribution
    fig, ax = plt.subplots()
    sns.countplot(x=monthly_data['anomaly'], ax=ax)
    ax.set_title("Anomalies Detected")
    ax.set_xticklabels(["Normal", "Anomalous"])
    st.pyplot(fig)

# Theft Prediction
elif option == "Theft Prediction":
    st.title("Electricity Theft Prediction with Random Forest")
    
    # Define features and labels
    final_data = data.copy()
    
    # Create the 'theft_label' based on the consumption comparison
    final_data['theft_label'] = final_data.apply(lambda row: 1 if row['daily_consumption'] < 0.5 * row['base_consumption'] else 0, axis=1)
    
    # Create a 'month' column for temporal information
    final_data['month'] = final_data['date'].dt.year * 100 + final_data['date'].dt.month
    
    # Retain 'household_id' for later use (predictions)
    household_ids = final_data['household_id']
    
    # Drop columns that are not needed for the model
    X = final_data.drop(columns=['theft_label', 'date', 'household_id'])  # Drop 'household_id'
    y = final_data['theft_label']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest model
    rf_clf = RandomForestClassifier(random_state=42, n_estimators=100)
    rf_clf.fit(X_train, y_train)
    y_pred = rf_clf.predict(X_test)
    
    # Evaluate model performance
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    st.pyplot(fig)
    
    # Evaluate model with Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_test, y_pred)
    st.write(f"Mean Absolute Error (MAE): {mae}")
    
    # Predict on the full dataset (to check which households are likely to steal electricity)
    final_data['predicted_theft'] = rf_clf.predict(X)
    
    # Filter and display households predicted to be involved in electricity theft
    theft_households = final_data[final_data['predicted_theft'] == 1][['household_id']].drop_duplicates()
    
    st.write("Households predicted to be involved in electricity theft:")
    st.write(theft_households)

# Daily Consumption Trends
elif option == "Daily Consumption Trends":
    st.title("Daily Consumption Trends Over Time")
    
    data['year'] = data['date'].dt.year
    fig, ax = plt.subplots()
    sns.lineplot(data=data, x='date', y='daily_consumption', hue='year', ax=ax)
    ax.set_title("Daily Consumption Trends Per Year")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Consumption (kWh)")
    st.pyplot(fig)

st.sidebar.write("Upload the dataset to Streamlit and run 'streamlit run app.py' to see results interactively!")
