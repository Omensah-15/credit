import pandas as pd
import numpy as np

def preprocess_data(df):
    """
    Preprocess data for training (same as inference preprocessing)
    """
    # Convert categorical variables
    employment_mapping = {'employed': 0, 'self-employed': 1, 'unemployed': 2, 'student': 3}
    education_mapping = {'High School': 0, 'Diploma': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4}
    loan_purpose_mapping = {
        'Business': 0, 'Crypto-Backed': 1, 'Car Loan': 2, 'Education': 3, 'Home Loan': 4
    }
    
    df = df.copy()
    
    df['employment_status'] = df['employment_status'].map(employment_mapping)
    df['education_level'] = df['education_level'].map(education_mapping)
    df['loan_purpose'] = df['loan_purpose'].map(loan_purpose_mapping)
    df['collateral_present'] = df['collateral_present'].map({'Yes': 1, 'No': 0})
    
    # Feature engineering
    df['income_to_loan_ratio'] = df['annual_income'] / (df['loan_amount'] + 1)
    df['credit_utilization'] = df['num_previous_loans'] / (df['credit_history_length'] + 1)
    df['default_rate'] = df['num_defaults'] / (df['num_previous_loans'] + 1)
    
    return df

def preprocess_inference_data(input_data):
    """
    Preprocess input data for inference (same as training preprocessing)
    """
    # Convert categorical variables
    employment_mapping = {'employed': 0, 'self-employed': 1, 'unemployed': 2, 'student': 3}
    education_mapping = {'High School': 0, 'Diploma': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4}
    loan_purpose_mapping = {
        'Business': 0, 'Crypto-Backed': 1, 'Car Loan': 2, 'Education': 3, 'Home Loan': 4
    }
    
    df = pd.DataFrame([input_data])
    
    df['employment_status'] = df['employment_status'].map(employment_mapping)
    df['education_level'] = df['education_level'].map(education_mapping)
    df['loan_purpose'] = df['loan_purpose'].map(loan_purpose_mapping)
    df['collateral_present'] = df['collateral_present'].map({'Yes': 1, 'No': 0})
    
    # Feature engineering
    df['income_to_loan_ratio'] = df['annual_income'] / (df['loan_amount'] + 1)
    df['credit_utilization'] = df['num_previous_loans'] / (df['credit_history_length'] + 1)
    df['default_rate'] = df['num_defaults'] / (df['num_previous_loans'] + 1)
    
    return df
