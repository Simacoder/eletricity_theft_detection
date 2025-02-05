import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv('synthetic_household_energy_data_with_metadata.csv', parse_dates=['date'])

data = load_data()

# Sidebar
st.sidebar.title("Energy Consumption Analysis")
option = st.sidebar.selectbox("Select Analysis", [
    "Dataset Overview", "Cluster Analysis", "Anomaly Detection", "Theft Prediction"])

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
    
    # Plot anomalies
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
    final_data['theft_label'] = final_data.apply(lambda row: 1 if row['daily_consumption'] < 0.5 * row['base_consumption'] else 0, axis=1)
    final_data['month'] = final_data['date'].dt.year * 100 + final_data['date'].dt.month
    X = final_data.drop(columns=['theft_label', 'household_id', 'date'])
    y = final_data['theft_label']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    rf_clf = RandomForestClassifier(random_state=42, n_estimators=100)
    rf_clf.fit(X_train, y_train)
    y_pred = rf_clf.predict(X_test)
    
    # Display results
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    st.pyplot(fig)
    
    # Display households with high likelihood of theft
    high_theft_households = final_data[final_data['theft_label'] == 1]['household_id'].unique()
    st.write("Households with High Likelihood of Electricity Theft:")
    st.write(high_theft_households)

st.sidebar.write("Upload the dataset to Streamlit and run 'streamlit run app.py' to see results interactively!")
