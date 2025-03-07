import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point

# ML & Evaluation
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Streamlit Page Configuration
st.set_page_config(page_title="Smart Meter Fraud Detection", layout="wide")

# 📌 Title
st.title("🔍 Smart Meter Fraud Detection & Analysis")

# 📌 Sidebar - Upload Dataset
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Convert Timestamp to DateTime format
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # Extract Time Features
    df['Hour'] = df['Timestamp'].dt.hour
    df['Day'] = df['Timestamp'].dt.day
    df['Month'] = df['Timestamp'].dt.month
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek

    # Select Features & Target
    features = ['Hour', 'Day', 'Month', 'DayOfWeek', 'Energy Consumption (kWh)', 'Voltage (kV)', 'Frequency (Hz)', 'Power Factor']
    X = df[features]
    y = df['Fraud']

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 📌 Sidebar - Select Model
    st.sidebar.header("Select Machine Learning Model")
    model_choice = st.sidebar.selectbox(
        "Choose Model",
        ["Random Forest", "Isolation Forest", "Logistic Regression", "XGBoost"]
    )

    # Model Initialization
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Isolation Forest": IsolationForest(contamination=0.1, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=500),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    # Train & Evaluate Model
    if st.sidebar.button("Train Model"):
        model = models[model_choice]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Compute Evaluation Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="macro")
        recall = recall_score(y_test, y_pred, average="macro")
        f1 = f1_score(y_test, y_pred, average="macro")

        # **Fix for IsolationForest: Use decision_function instead of predict_proba**
        if model_choice == "Isolation Forest":
            roc_auc = roc_auc_score(y_test, model.decision_function(X_test))
        else:
            roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

        # 📊 Display Performance Metrics
        st.subheader("📊 Model Performance")
        st.write(f"**Model:** {model_choice}")
        st.write(f"**Accuracy:** {accuracy:.4f}")
        st.write(f"**Precision:** {precision:.4f}")
        st.write(f"**Recall:** {recall:.4f}")
        st.write(f"**F1 Score:** {f1:.4f}")
        st.write(f"**ROC-AUC:** {roc_auc:.4f}")

        # 🔍 Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix - {model_choice}")
        st.pyplot(fig)

    # 📍 Fraud Risk Mapping
    if st.checkbox("Show Fraud Risk Map"):
        st.subheader("📍 Fraud Risk Mapping (Geospatial Analysis)")

        # Filter Fraud Cases
        fraud_data = df[df['Fraud'] == 1]

        # Convert to GeoDataFrame
        geometry = [Point(xy) for xy in zip(fraud_data['Longitude'], fraud_data['Latitude'])]
        gdf = gpd.GeoDataFrame(fraud_data, geometry=geometry)

        # **Fix for GeoPandas: Use a manually downloaded map**
        sa_map = gpd.read_file("zaf_adm_sadb_ocha_20201109_SHP/zaf_admbnda_adm1_sadb_ocha_20201109.shp")  # Download from Natural Earth

        # Plot High-Risk Fraud Areas
        fig, ax = plt.subplots(figsize=(8, 6))
        sa_map.plot(ax=ax, color='lightgrey')
        gdf.plot(ax=ax, marker='o', color='red', markersize=5)
        plt.title("📍 High-Risk Fraud Areas in South Africa")
        st.pyplot(fig)

    # 📈 Fraudulent Consumption Trend Analysis
    if st.checkbox("Show Consumption Trend Analysis"):
        st.subheader("📈 Fraudulent vs Normal Consumption Trend")

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(data=df[df['Fraud'] == 1], x="Hour", y="Energy Consumption (kWh)", ci=None, label="Fraudulent", ax=ax)
        sns.lineplot(data=df[df['Fraud'] == 0], x="Hour", y="Energy Consumption (kWh)", ci=None, label="Normal", ax=ax)
        plt.xlabel("Hour of the Day")
        plt.ylabel("Energy Consumption (kWh)")
        plt.title("📈 Energy Consumption Trends")
        plt.legend()
        st.pyplot(fig)

    # 📊 Model Comparison
    if st.checkbox("Show Model Performance Comparison"):
        st.subheader("📊 Model Performance Comparison")

        # Compare Models
        performance = []
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average="macro")
            recall = recall_score(y_test, y_pred, average="macro")
            f1 = f1_score(y_test, y_pred, average="macro")

            # **Fix for IsolationForest: Use decision_function**
            if name == "Isolation Forest":
                roc_auc = roc_auc_score(y_test, model.decision_function(X_test))
            else:
                roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

            performance.append({
                "Model": name,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1,
                "ROC-AUC": roc_auc
            })

        # Convert to DataFrame
        performance_df = pd.DataFrame(performance)
        st.write(performance_df)

        # 📊 Plot Performance Trends
        fig, ax = plt.subplots(figsize=(10, 5))
        for metric in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]:
            ax.plot(performance_df["Model"], performance_df[metric], marker='o', label=metric)
        plt.xlabel("Model")
        plt.ylabel("Score")
        plt.title("📊 Model Performance Trends")
        plt.legend()
        plt.xticks(rotation=15)
        plt.grid()
        st.pyplot(fig)

# Footer
st.sidebar.markdown("👨‍💻 **Authors:** Simanga Mchunu, Nkosinathi Nhlapo, Kagiso Leboka, Bongani Baloyi")
st.sidebar.markdown("📅 **Project:** Smart Meter Fraud Detection")
