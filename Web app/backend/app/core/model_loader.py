import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from typing import Any

MODEL_PATH = os.path.join(os.getcwd(), "fraud_detection_model.pkl")

def load_model() -> Any:

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    
    model = joblib.load(MODEL_PATH)
    return model

def predict_fraud(model: Any, features: list) -> int:
       
    prediction = model.predict([features])
    return prediction[0]

def get_fraud_detection_model() -> Any:

    model = load_model()
    return model
