import pandas as pd
import numpy as np
import os

# Create directories
os.makedirs('data', exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# Create sample data
n_samples = 1000

data = {
    'customer_id': [f'CUST{str(i).zfill(5)}' for i in range(1, n_samples + 1)],
    'age': np.random.randint(25, 70, n_samples),
    'employment_status': np.random.choice(['employed', 'self-employed', 'unemployed', 'student'], n_samples, p=[0.6, 0.2, 0.1, 0.1]),
    'annual_income': np.random.randint(20000, 100000, n_samples),
    'education_level': np.random.choice(['High School', 'Diploma', 'Bachelor', 'Master'], n_samples, p=[0.2, 0.3, 0.4, 0.1]),
    'credit_history_length': np.random.randint(1, 20, n_samples),
    'num_previous_loans': np.random.randint(0, 10, n_samples),
    'num_defaults': np.random.randint(0, 3, n_samples),
    'avg_payment_delay_days': np.random.randint(0, 15, n_samples),
    'current_credit_score': np.random.randint(500, 800, n_samples),
    'loan_amount': np.random.randint(5000, 150000, n_samples),
    'loan_term_months': np.random.choice([12, 24, 36, 48, 60], n_samples),
    'loan_purpose': np.random.choice(['Business', 'Crypto-Backed', 'Car Loan', 'Education', 'Home Loan'], n_samples),
    'collateral_present': np.random.choice(['Yes', 'No'], n_samples, p=[0.7, 0.3]),
    'identity_verified_on_chain': np.random.randint(0, 2, n_samples),
    'transaction_consistency_score': np.round(np.random.uniform(0.2, 1.0, n_samples), 2),
    'fraud_alert_flag': np.random.randint(0, 2, n_samples, p=[0.9, 0.1]),
    'on_chain_credit_history': np.random.randint(0, 10, n_samples),
}

# Create target variable with some logic to make it realistic
df = pd.DataFrame(data)

# Probability of default based on features
prob_default = (
    0.3 * (df['num_defaults'] > 0) +
    0.2 * (df['employment_status'] == 'unemployed') +
    0.1 * (df['current_credit_score'] < 600) +
    0.1 * (df['avg_payment_delay_days'] > 7) +
    0.1 * (df['loan_amount'] / df['annual_income'] > 0.5) +
    0.1 * (df['fraud_alert_flag'] == 1) +
    np.random.normal(0, 0.1, n_samples)
)

# Convert to binary outcome
df['default_flag'] = (prob_default > 0.5).astype(int)
df['probability_of_default'] = np.clip(1 / (1 + np.exp(-prob_default)), 0.01, 0.99)

# Save to CSV
df.to_csv('data/sample_dataset.csv', index=False)
print(f"Sample dataset with {n_samples} records created at data/sample_dataset.csv")
print(f"Default rate: {df['default_flag'].mean():.2%}")
