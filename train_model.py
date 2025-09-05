import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from utils.preprocessing import preprocess_data

# Create directories if they don't exist
os.makedirs('models', exist_ok=True)

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def objective(trial, X, y):
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'min_split_gain': trial.suggest_float('min_split_gain', 1e-8, 10.0, log=True),
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Apply SMOTE only to training data
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train_res, y_train_res)
        
        y_pred = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, y_pred)
        scores.append(score)
    
    return np.mean(scores)

def train_advanced_model():
    print("Loading data...")
    df = load_data('data/sample_dataset.csv')
    print("Preprocessing data...")
    df = preprocess_data(df)
    
    # Define features and target
    features = [
        'age', 'employment_status', 'annual_income', 'education_level', 
        'credit_history_length', 'num_previous_loans', 'num_defaults', 
        'avg_payment_delay_days', 'current_credit_score', 'loan_amount', 
        'loan_term_months', 'loan_purpose', 'collateral_present', 
        'identity_verified_on_chain', 'transaction_consistency_score', 
        'fraud_alert_flag', 'on_chain_credit_history', 'income_to_loan_ratio',
        'credit_utilization', 'default_rate'
    ]
    
    X = df[features]
    y = df['default_flag']
    
    # Save feature columns for inference
    joblib.dump(features, 'models/feature_columns.pkl')
    joblib.dump(X.columns.tolist(), 'models/model_columns.pkl')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Starting Optuna optimization...")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=50)
    
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Train final model with best parameters
    best_params = trial.params
    best_params['objective'] = 'binary'
    best_params['metric'] = 'binary_logloss'
    best_params['verbosity'] = -1
    best_params['boosting_type'] = 'gbdt'
    
    # Apply SMOTE to entire training set
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    print("Training final model...")
    final_model = lgb.LGBMClassifier(**best_params)
    final_model.fit(X_train_res, y_train_res)
    
    # Save the base model for SHAP explanations
    joblib.dump(final_model, 'models/trained_lgbm_model.pkl')
    
    # Calibrate the model
    print("Calibrating model...")
    calibrated_model = CalibratedClassifierCV(final_model, method='isotonic', cv=5)
    calibrated_model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred_proba = calibrated_model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test ROC AUC: {roc_auc:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save calibrated model
    joblib.dump(calibrated_model, 'models/calibration_model.pkl')
    
    print("Models saved successfully!")
    return calibrated_model, features, X_test, y_test

if __name__ == "__main__":
    train_advanced_model()
