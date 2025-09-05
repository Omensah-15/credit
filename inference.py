import joblib
import pandas as pd
import numpy as np
from utils.preprocessing import preprocess_inference_data

def load_models():
    """
    Load trained models and feature columns
    """
    try:
        model = joblib.load('models/calibration_model.pkl')
        feature_columns = joblib.load('models/feature_columns.pkl')
        return model, feature_columns
    except FileNotFoundError as e:
        raise Exception("Models not found. Please run train_model.py first.") from e

def predict_risk(input_data):
    """
    Predict risk for a single application
    """
    # Load models
    model, feature_columns = load_models()
    
    # Preprocess input data
    processed_data = preprocess_inference_data(input_data)
    
    # Ensure we have all required features
    for col in feature_columns:
        if col not in processed_data.columns:
            processed_data[col] = 0  # Add missing columns with default value
    
    # Select and order features correctly
    processed_data = processed_data[feature_columns]
    
    # Make prediction
    probability_of_default = model.predict_proba(processed_data)[0][1]
    
    # Convert to risk score (0-1000, higher is better)
    risk_score = int((1 - probability_of_default) * 1000)
    
    # Categorize risk
    if probability_of_default < 0.1:
        risk_category = "Very Low Risk"
    elif probability_of_default < 0.2:
        risk_category = "Low Risk"
    elif probability_of_default < 0.4:
        risk_category = "Medium Risk"
    elif probability_of_default < 0.6:
        risk_category = "High Risk"
    else:
        risk_category = "Very High Risk"
    
    return probability_of_default, risk_score, risk_category, processed_data
