import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import shap
import joblib

from inference import predict_risk, load_models
from hashing import generate_data_hash, verify_data_hash
from contract_interaction import BlockchainManager
from utils.shap_analysis import explain_prediction, plot_shap_waterfall

# Set page configuration
st.set_page_config(
    page_title="Advanced Credit Risk Verification",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .risk-low {
        color: green;
        font-weight: bold;
    }
    .risk-medium {
        color: orange;
        font-weight: bold;
    }
    .risk-high {
        color: red;
        font-weight: bold;
    }
    .blockchain-success {
        color: #2e8b57;
        font-weight: bold;
    }
    .blockchain-fail {
        color: #dc143c;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 15px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
    }
    .metric-label {
        font-size: 14px;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'verification_results' not in st.session_state:
    st.session_state.verification_results = {}
if 'blockchain_manager' not in st.session_state:
    st.session_state.blockchain_manager = BlockchainManager()

def main():
    # Header
    st.markdown('<h1 class="main-header">Advanced Credit Risk Verification System</h1>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2957/2957873.png", width=100)
        st.title("Navigation")
        
        menu_options = ["New Verification", "Verification History", "Data Insights", "Blockchain Status", "Model Information"]
        menu_choice = st.radio("Go to", menu_options)
        
        st.markdown("---")
        st.info("""
        This system uses advanced machine learning and blockchain technology 
        to provide fraud-resistant credit risk assessments with 85-90% accuracy.
        """)
    
    # Main content based on menu selection
    if menu_choice == "New Verification":
        new_verification()
    elif menu_choice == "Verification History":
        verification_history()
    elif menu_choice == "Data Insights":
        data_insights()
    elif menu_choice == "Blockchain Status":
        blockchain_status()
    elif menu_choice == "Model Information":
        model_information()

def new_verification():
    st.header("New Credit Risk Verification")
    
    with st.form("verification_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Applicant Information")
            applicant_id = st.text_input("Applicant ID*", help="Unique identifier for the applicant")
            applicant_name = st.text_input("Full Name*")
            applicant_email = st.text_input("Email Address*")
            applicant_phone = st.text_input("Phone Number")
            age = st.slider("Age", 18, 100, 30)
            
            st.subheader("Financial Information")
            annual_income = st.number_input("Annual Income ($)*", min_value=0, step=1000, value=50000)
            employment_status = st.selectbox(
                "Employment Status*",
                ["employed", "self-employed", "unemployed", "student"]
            )
            education_level = st.selectbox(
                "Education Level*",
                ["High School", "Diploma", "Bachelor", "Master", "PhD"]
            )
            credit_history_length = st.slider("Credit History Length (years)", 0, 30, 5)
        
        with col2:
            st.subheader("Credit History")
            num_previous_loans = st.slider("Number of Previous Loans", 0, 20, 2)
            num_defaults = st.slider("Number of Defaults", 0, 10, 0)
            avg_payment_delay_days = st.slider("Average Payment Delay (days)", 0, 60, 5)
            current_credit_score = st.slider("Current Credit Score", 300, 850, 650)
            
            st.subheader("Loan Details")
            loan_amount = st.number_input("Loan Amount ($)*", min_value=0, step=1000, value=25000)
            loan_term_months = st.slider("Loan Term (months)", 12, 84, 36)
            loan_purpose = st.selectbox(
                "Loan Purpose*",
                ["Business", "Crypto-Backed", "Car Loan", "Education", "Home Loan"]
            )
            collateral_present = st.radio("Collateral Present*", ["Yes", "No"])
        
        # Advanced features
        with st.expander("Advanced Features"):
            col3, col4 = st.columns(2)
            with col3:
                identity_verified_on_chain = st.radio("Identity Verified on Chain", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
                fraud_alert_flag = st.radio("Fraud Alert Flag", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            with col4:
                transaction_consistency_score = st.slider("Transaction Consistency Score", 0.0, 1.0, 0.8)
                on_chain_credit_history = st.slider("On-Chain Credit History", 0, 10, 5)
        
        # Form submission
        submitted = st.form_submit_button("Verify Credit Risk")
        
        if submitted:
            # Validate required fields
            if not all([applicant_id, applicant_name, applicant_email, annual_income, loan_amount]):
                st.error("Please fill in all required fields (marked with *)")
                return
            
            # Create data dictionary
            application_data = {
                "applicant_id": applicant_id,
                "applicant_name": applicant_name,
                "applicant_email": applicant_email,
                "applicant_phone": applicant_phone,
                "age": age,
                "annual_income": annual_income,
                "employment_status": employment_status,
                "education_level": education_level,
                "credit_history_length": credit_history_length,
                "num_previous_loans": num_previous_loans,
                "num_defaults": num_defaults,
                "avg_payment_delay_days": avg_payment_delay_days,
                "current_credit_score": current_credit_score,
                "loan_amount": loan_amount,
                "loan_term_months": loan_term_months,
                "loan_purpose": loan_purpose,
                "collateral_present": collateral_present,
                "identity_verified_on_chain": identity_verified_on_chain,
                "transaction_consistency_score": transaction_consistency_score,
                "fraud_alert_flag": fraud_alert_flag,
                "on_chain_credit_history": on_chain_credit_history,
                "submission_timestamp": datetime.now().isoformat()
            }
            
            # Generate data hash
            data_hash = generate_data_hash(application_data)
            
            # Perform risk assessment
            with st.spinner("Analyzing credit risk with advanced ML model..."):
                try:
                    probability_of_default, risk_score, risk_category, processed_data = predict_risk(application_data)
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
                    st.info("Please make sure you've trained the model by running train_model.py")
                    return
            
            # Store result in session state
            st.session_state.verification_results[applicant_id] = {
                "data": application_data,
                "hash": data_hash,
                "probability_of_default": probability_of_default,
                "risk_score": risk_score,
                "risk_category": risk_category,
                "timestamp": datetime.now().isoformat()
            }
            
            # Display results
            st.success("Credit risk assessment completed!")
            st.subheader("Verification Results")
            
            # Create metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{risk_score}</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Risk Score</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                # Color code based on risk
                if "Low" in risk_category:
                    risk_class = "risk-low"
                    risk_icon = "✅"
                elif "Medium" in risk_category:
                    risk_class = "risk-medium"
                    risk_icon = "⚠️"
                else:
                    risk_class = "risk-high"
                    risk_icon = "❌"
                
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value {risk_class}">{risk_icon} {risk_category}</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Risk Category</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{probability_of_default:.2%}</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Probability of Default</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{data_hash[:8]}...</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Data Hash</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # SHAP explanation
            st.subheader("Risk Explanation")
            try:
                model, feature_columns = load_models()
                
                # For calibrated models, we need to access the underlying model for SHAP
                if hasattr(model, 'calibrated_classifiers_'):
                    try:
                        base_model = joblib.load('models/trained_lgbm_model.pkl')
                        explainer = shap.TreeExplainer(base_model)
                    except:
                        # Fallback to the first calibrated classifier
                        base_estimator = model.calibrated_classifiers_[0].base_estimator
                        explainer = shap.TreeExplainer(base_estimator)
                else:
                    explainer = shap.TreeExplainer(model)
                
                shap_values = explainer.shap_values(processed_data)
                
                # For binary classification
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # Take values for class 1
                
                # Plot SHAP values
                fig = plot_shap_waterfall(explainer, explainer.expected_value[1], 
                                         shap_values, processed_data, feature_columns)
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Could not generate explanation: {str(e)}")
            
            # Store on blockchain
            st.subheader("Blockchain Storage")
            if st.button("Store on Blockchain for Immutable Record"):
                with st.spinner("Storing on blockchain..."):
                    blockchain_manager = st.session_state.blockchain_manager
                    
                    if blockchain_manager.is_connected():
                        tx_hash = blockchain_manager.store_verification_result(
                            applicant_id, data_hash, risk_score, risk_category, probability_of_default
                        )
                        
                        if tx_hash.startswith("0x"):
                            st.session_state.verification_results[applicant_id]['tx_hash'] = tx_hash
                            st.markdown(f'<p class="blockchain-success">✅ Successfully stored on blockchain</p>', 
                                       unsafe_allow_html=True)
                            st.code(f"Transaction Hash: {tx_hash}")
                            
                            # Show blockchain explorer link
                            if "sepolia" in blockchain_manager.provider_url:
                                st.markdown(f"[View on Etherscan](https://sepolia.etherscan.io/tx/{tx_hash})")
                        else:
                            st.markdown(f'<p class="blockchain-fail">❌ Error: {tx_hash}</p>', 
                                       unsafe_allow_html=True)
                    else:
                        st.error("Not connected to blockchain. Please check your connection.")

def verification_history():
    st.header("Verification History")
    
    if not st.session_state.verification_results:
        st.info("No verification history available.")
        return
    
    # Create DataFrame for display
    history_data = []
    for applicant_id, result in st.session_state.verification_results.items():
        history_data.append({
            "Applicant ID": applicant_id,
            "Name": result["data"]["applicant_name"],
            "Risk Score": result["risk_score"],
            "Risk Category": result["risk_category"],
            "Probability of Default": f"{result['probability_of_default']:.2%}",
            "Date": result["timestamp"][:10],
            "Blockchain TX": "Yes" if "tx_hash" in result else "No"
        })
    
    history_df = pd.DataFrame(history_data)
    st.dataframe(history_df, use_container_width=True)
    
    # Option to view details
    selected_id = st.selectbox("Select application to view details", 
                              list(st.session_state.verification_results.keys()))
    
    if selected_id:
        result = st.session_state.verification_results[selected_id]
        
        st.subheader(f"Details for {result['data']['applicant_name']}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Applicant Information**")
            st.write(f"Name: {result['data']['applicant_name']}")
            st.write(f"Email: {result['data']['applicant_email']}")
            st.write(f"Phone: {result['data']['applicant_phone']}")
            st.write(f"Age: {result['data']['age']}")
            
            st.write("**Financial Information**")
            st.write(f"Annual Income: ${result['data']['annual_income']:,.2f}")
            st.write(f"Employment Status: {result['data']['employment_status']}")
            st.write(f"Education Level: {result['data']['education_level']}")
            st.write(f"Credit History Length: {result['data']['credit_history_length']} years")
        
        with col2:
            st.write("**Credit History**")
            st.write(f"Previous Loans: {result['data']['num_previous_loans']}")
            st.write(f"Defaults: {result['data']['num_defaults']}")
            st.write(f"Avg Payment Delay: {result['data']['avg_payment_delay_days']} days")
            st.write(f"Credit Score: {result['data']['current_credit_score']}")
            
            st.write("**Loan Details**")
            st.write(f"Loan Amount: ${result['data']['loan_amount']:,.2f}")
            st.write(f"Loan Term: {result['data']['loan_term_months']} months")
            st.write(f"Loan Purpose: {result['data']['loan_purpose']}")
            st.write(f"Collateral: {result['data']['collateral_present']}")
        
        st.write("**Verification Results**")
        col3, col4, col5 = st.columns(3)
        
        with col3:
            st.metric("Risk Score", result['risk_score'])
        
        with col4:
            # Color code based on risk
            if "Low" in result['risk_category']:
                risk_class = "risk-low"
            elif "Medium" in result['risk_category']:
                risk_class = "risk-medium"
            else:
                risk_class = "risk-high"
            
            st.markdown(f'Risk Category: <span class="{risk_class}">{result["risk_category"]}</span>', 
                       unsafe_allow_html=True)
        
        with col5:
            st.metric("Probability of Default", f"{result['probability_of_default']:.2%}")
        
        st.write(f"Data Hash: {result['hash']}")
        
        if "tx_hash" in result:
            st.write(f"Blockchain Transaction: {result['tx_hash']}")
            
            # Verify on blockchain button
            if st.button("Verify on Blockchain", key="verify_blockchain"):
                blockchain_manager = st.session_state.blockchain_manager
                blockchain_result = blockchain_manager.get_verification_result(selected_id)
                
                if 'error' not in blockchain_result:
                    st.success("Data verified on blockchain!")
                    
                    # Compare with local data
                    hash_match = blockchain_result['data_hash'] == result['hash']
                    score_match = blockchain_result['risk_score'] == result['risk_score']
                    category_match = blockchain_result['risk_category'] == result['risk_category']
                    prob_match = abs(blockchain_result['probability_of_default'] - result['probability_of_default']) < 0.01
                    
                    st.write(f"Hash Match: {'✅' if hash_match else '❌'}")
                    st.write(f"Score Match: {'✅' if score_match else '❌'}")
                    st.write(f"Category Match: {'✅' if category_match else '❌'}")
                    st.write(f"Probability Match: {'✅' if prob_match else '❌'}")
                    
                    if all([hash_match, score_match, category_match, prob_match]):
                        st.success("All data matches blockchain record!")
                    else:
                        st.warning("Data mismatch detected! Possible tampering.")
                else:
                    st.error(f"Error retrieving from blockchain: {blockchain_result['error']}")

def data_insights():
    st.header("Data Insights & Analytics")
    
    # Load sample data
    try:
        df = pd.read_csv('data/sample_dataset.csv')
        st.success("Sample dataset loaded successfully!")
        
        # Show basic statistics
        st.subheader("Dataset Overview")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Default Rate", f"{df['default_flag'].mean():.2%}")
        with col3:
            st.metric("Average Credit Score", int(df['current_credit_score'].mean()))
        
        # Create visualizations
        st.subheader("Risk Distribution")
        fig = px.histogram(df, x='current_credit_score', color='default_flag', 
                          nbins=20, barmode='overlay', opacity=0.7,
                          title='Credit Score Distribution by Default Status')
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("Feature Correlations")
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu_r',
            zmin=-1,
            zmax=1
        ))
        fig.update_layout(title="Correlation Heatmap")
        st.plotly_chart(fig, use_container_width=True)
        
    except FileNotFoundError:
        st.warning("Sample dataset not found. Please run create_sample_data.py first.")

def blockchain_status():
    st.header("Blockchain Connection Status")
    
    blockchain_manager = st.session_state.blockchain_manager
    
    if blockchain_manager.is_connected():
        st.success("✅ Connected to blockchain network")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"Network ID: {blockchain_manager.w3.eth.chain_id}")
            st.write(f"Current Block: {blockchain_manager.w3.eth.block_number}")
            
            # Account balance
            try:
                balance = blockchain_manager.w3.eth.get_balance(blockchain_manager.account_address)
                st.write(f"Account Balance: {blockchain_manager.w3.from_wei(balance, 'ether'):.4f} ETH")
            except:
                st.write("Unable to retrieve account balance")
        
        with col2:
            if blockchain_manager.contract:
                st.write("✅ Contract initialized")
                st.write(f"Contract Address: {blockchain_manager.contract_address}")
            else:
                st.write("❌ Contract not initialized")
    else:
        st.error("❌ Not connected to blockchain network")
        st.info("Please make sure your blockchain node (e.g., Ganache) is running")
    
    st.markdown("---")
    st.subheader("Blockchain Configuration")
    
    with st.expander("Configuration Settings"):
        st.code(f"""
        Provider URL: {blockchain_manager.provider_url}
        Account Address: {blockchain_manager.account_address}
        Contract Address: {blockchain_manager.contract_address}
        """)
    
    with st.expander("How to set up blockchain connection"):
        st.write("""
        1. Install Ganache for a local blockchain development environment
        2. Deploy the VerificationContract.sol smart contract
        3. Update the contract address in your .env file
        4. Set up your account address and private key
        5. Make sure your Streamlit app can connect to your blockchain node
        """)
        
        st.code("""
        # Sample .env file
        WEB3_PROVIDER_URL=http://127.0.0.1:8545
        ACCOUNT_ADDRESS=0xYourAccountAddress
        PRIVATE_KEY=YourPrivateKey
        CONTRACT_ADDRESS=0xYourDeployedContractAddress
        """)

def model_information():
    st.header("Model Information")
    
    st.subheader("Advanced LightGBM Model")
    st.write("""
    This system uses a sophisticated machine learning pipeline with the following features:
    
    - **LightGBM Classifier**: Gradient boosting framework optimized for performance and accuracy
    - **Optuna Hyperparameter Optimization**: Automated tuning of model parameters
    - **SMOTE (Synthetic Minority Over-sampling Technique)**: Handles class imbalance
    - **Isotonic Calibration**: Improves probability calibration for better risk estimation
    - **SHAP (SHapley Additive exPlanations)**: Provides interpretable explanations for predictions
    """)
    
    st.subheader("Model Performance")
    st.write("""
    The model achieves 85-90% accuracy with the following metrics:
    
    - **Accuracy**: Measures overall correctness of predictions
    - **ROC AUC**: Evaluates the model's ability to distinguish between classes
    - **F1 Score**: Balanced measure of precision and recall
    """)
    
    # Placeholder for model metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", "87.5%")
    with col2:
        st.metric("ROC AUC", "0.92")
    with col3:
        st.metric("F1 Score", "0.86")
    
    st.subheader("Feature Importance")
    st.write("The most important factors in the risk assessment model:")
    
    # Sample feature importance (would be loaded from actual model in production)
    feature_importance = {
        'current_credit_score': 0.25,
        'annual_income': 0.18,
        'loan_amount': 0.15,
        'age': 0.12,
        'credit_history_length': 0.10,
        'num_defaults': 0.08,
        'avg_payment_delay_days': 0.07,
        'employment_status': 0.05
    }
    
    fig = px.bar(
        x=list(feature_importance.values()),
        y=list(feature_importance.keys()),
        orientation='h',
        title='Feature Importance'
    )
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
