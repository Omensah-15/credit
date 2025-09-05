import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt

def explain_prediction(model, input_data, feature_names):
    """
    Generate SHAP explanation for a prediction
    """
    # For calibrated models, we need to access the underlying model
    if hasattr(model, 'calibrated_classifiers_'):
        # Load the base model for SHAP explanations
        try:
            base_model = joblib.load('models/trained_lgbm_model.pkl')
            explainer = shap.TreeExplainer(base_model)
        except:
            # Fallback to the first calibrated classifier
            base_estimator = model.calibrated_classifiers_[0].base_estimator
            explainer = shap.TreeExplainer(base_estimator)
    else:
        # Regular model
        explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(input_data)
    
    # For binary classification
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Take values for class 1
    
    return explainer, shap_values

def plot_shap_summary(shap_values, features, feature_names, max_display=20):
    """
    Plot SHAP summary plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, features, feature_names=feature_names, 
                      max_display=max_display, show=False)
    plt.tight_layout()
    return fig

def plot_shap_waterfall(explainer, expected_value, shap_values, features, feature_names, index=0):
    """
    Plot SHAP waterfall plot for a single prediction
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.decision_plot(expected_value, shap_values, features.iloc[index], 
                       feature_names=feature_names, show=False)
    plt.tight_layout()
    return fig
