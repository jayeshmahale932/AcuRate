import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="AcuRate - Interest Rate Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 4.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
        letter-spacing: 2px;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #555;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: 500;
    }
    .description {
        font-size: 1.1rem;
        color: #777;
        text-align: center;
        margin-bottom: 3rem;
        font-style: italic;
    }
    .prediction-box {
        background-color: #f0f8ff;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        border: 2px solid #1f77b4;
    }
    .prediction-value {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .info-box {
        background-color: #f9f9f9;
        border-left: 5px solid #1f77b4;
        padding: 10px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Load models
@st.cache_resource
def load_models():
    """Load all trained models"""
    models = {}
    model_path = PROJECT_ROOT / "saved_models"
    
    try:
        models['Linear Regression'] = joblib.load(model_path / "linear_model.joblib")
        models['Decision Tree'] = joblib.load(model_path / "dtree_model.joblib")
        models['Random Forest'] = joblib.load(model_path / "rf_model.joblib")
        models['Gradient Boosting'] = joblib.load(model_path / "gbr_model.joblib")
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None
    
    return models

@st.cache_resource
def load_scaler():
    """Load the MinMaxScaler used during training"""
    try:
        scaler = joblib.load(PROJECT_ROOT / "saved_models" / "minmax_scaler.joblib")
        return scaler
    except Exception as e:
        st.error(f"Error loading scaler: {e}")
        return None

# Define feature columns (based on the feature engineered dataset)
FEATURE_COLUMNS = [
    'loan_amnt', 'term', 'grade', 'annual_inc', 'dti', 'delinq_2yrs',
    'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util',
    'total_acc', 'initial_list_status', 'open_rv_12m', 'open_rv_24m',
    'inc_loan_ratio', 'fico_score', 'verification_status_Source Verified',
    'verification_status_Verified', 'purpose_credit_card',
    'purpose_debt_consolidation', 'purpose_home_improvement', 'purpose_house',
    'purpose_major_purchase', 'purpose_medical', 'purpose_moving',
    'purpose_other', 'purpose_small_business', 'purpose_vacation'
]

numeric_features = ['loan_amnt', 'annual_inc', 'dti', 'delinq_2yrs', 'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'open_rv_12m', 'open_rv_24m', 'inc_loan_ratio', 'fico_score', 'int_rate']


def create_input_features(user_inputs):
    """Create feature vector from user inputs"""
    # Initialize feature dictionary with zeros
    features = {col: 0.0 for col in FEATURE_COLUMNS}
    
    # Fill in the user-provided values
    features['loan_amnt'] = user_inputs['loan_amnt']
    features['term'] = user_inputs['term']
    features['grade'] = user_inputs['grade']
    features['annual_inc'] = user_inputs['annual_inc']
    features['dti'] = user_inputs['dti']
    features['delinq_2yrs'] = user_inputs['delinq_2yrs']
    features['inq_last_6mths'] = user_inputs['inq_last_6mths']
    features['open_acc'] = user_inputs['open_acc']
    features['pub_rec'] = user_inputs['pub_rec']
    features['revol_bal'] = user_inputs['revol_bal']
    features['revol_util'] = user_inputs['revol_util']
    features['total_acc'] = user_inputs['total_acc']
    features['initial_list_status'] = user_inputs['initial_list_status']
    features['open_rv_12m'] = user_inputs['open_rv_12m']
    features['open_rv_24m'] = user_inputs['open_rv_24m']
    features['fico_score'] = user_inputs['fico_score']
    
    # Calculate derived features automatically
    features['inc_loan_ratio'] = user_inputs['annual_inc'] / user_inputs['loan_amnt'] if user_inputs['loan_amnt'] > 0 else 0
    
    # Set verification status (one-hot encoded) BEFORE creating DataFrame
    if user_inputs['verification_status'] == 'Source Verified':
        features['verification_status_Source Verified'] = 1
        features['verification_status_Verified'] = 0
    elif user_inputs['verification_status'] == 'Verified':
        features['verification_status_Source Verified'] = 0
        features['verification_status_Verified'] = 1
    else:  # Not Verified
        features['verification_status_Source Verified'] = 0
        features['verification_status_Verified'] = 0
    
    # Set purpose (one-hot encoded) BEFORE creating DataFrame
    purpose_key = f"purpose_{user_inputs['purpose'].lower().replace(' ', '_')}"
    if purpose_key in features:
        features[purpose_key] = 1
    
    # Create DataFrame with all features
    df = pd.DataFrame([features])

    scaler = load_scaler()
    df['int_rate'] = 0
    df[numeric_features] = scaler.transform(df[numeric_features])
    df.drop(columns=['int_rate'], inplace=True)
    return df


def main():
    # Header with icon
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('''
            <div style="text-align: center;">
                <svg xmlns="http://www.w3.org/2000/svg" width="120" height="120" viewBox="0 0 24 24" fill="none" stroke="#1f77b4" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <rect x="2" y="5" width="20" height="14" rx="2"/>
                    <line x1="2" y1="10" x2="22" y2="10"/>
                </svg>
            </div>
        ''', unsafe_allow_html=True)
    
    st.markdown('<p class="main-header">AcuRate</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Personalized Interest Rate Prediction System</p>', unsafe_allow_html=True)
    st.markdown('<p class="description">Predict loan interest rates using advanced machine learning models trained on borrower credit profiles</p>', unsafe_allow_html=True)
    
    # Load models
    models = load_models()
    if models is None:
        st.error("Failed to load models. Please ensure model files are in the 'saved_models' directory.")
        return
    
    # Sidebar for model selection and information
    with st.sidebar:
        st.markdown('''
            <div style="text-align: center; margin-bottom: 20px;">
                <svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="#1f77b4" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="12" cy="12" r="3"/>
                    <path d="M12 1v6m0 6v6m-8-7h6m6 0h6"/>
                    <circle cx="12" cy="12" r="10"/>
                </svg>
            </div>
        ''', unsafe_allow_html=True)
        st.title("Configuration")
        
        selected_model = st.selectbox(
            "Select Prediction Model",
            options=list(models.keys()),
            index=2  # Default to Random Forest
        )
        
        st.markdown("---")
        st.markdown('''
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#1f77b4" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
            </svg>
        ''', unsafe_allow_html=True)
        st.subheader("Model Information")
        model_info = {
            'Linear Regression': 'Fast, interpretable baseline model',
            'Decision Tree': 'Non-linear model with clear decision rules',
            'Random Forest': 'Ensemble model with high accuracy (Recommended)',
            'Gradient Boosting': 'Advanced ensemble with sequential learning'
        }
        st.info(model_info[selected_model])
        
        st.markdown("---")
        st.markdown('''
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#1f77b4" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="10"/>
                <line x1="12" y1="16" x2="12" y2="12"/>
                <line x1="12" y1="8" x2="12.01" y2="8"/>
            </svg>
        ''', unsafe_allow_html=True)
        st.subheader("About AcuRate")
        st.write("""
        This application predicts the interest rate a borrower is likely to be 
        charged based on their financial and credit profile using machine learning 
        models trained on Lending Club data.
        
        **Key Features:**
        - Multiple ML models for comparison
        - Real-time predictions
        - Comprehensive credit analysis
        - Financial calculations
        """)
    
    # Create tabs for better organization
    tab1, tab2, tab3 = st.tabs([
        "Loan Details", 
        "Credit Profile", 
        "Predict & Analyze"
    ])
    
    with tab1:
        st.markdown('''
            <div style="margin-bottom: 20px;">
                <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#1f77b4" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display: inline-block; vertical-align: middle; margin-right: 10px;">
                    <line x1="12" y1="1" x2="12" y2="23"/>
                    <path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>
                </svg>
                <h3 style="display: inline-block; vertical-align: middle; margin: 0;">Loan Information</h3>
            </div>
        ''', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            loan_amnt = st.number_input(
                "Loan Amount (₹)",
                min_value=40000,
                max_value=2500000,
                value=1000000,
                step=10000,
                help="The amount of loan requested by the borrower (typical range: ₹40,000 - ₹25,00,000)"
            )
            
            term = st.selectbox(
                "Loan Term (months)",
                options=[36, 60],
                index=0,
                help="The number of months for loan repayment"
            )
            
            purpose = st.selectbox(
                "Loan Purpose",
                options=['Credit Card', 'Debt Consolidation', 'Home Improvement', 
                        'House', 'Major Purchase', 'Medical', 'Moving', 
                        'Other', 'Small Business', 'Vacation'],
                index=1,
                help="The purpose for which the loan is requested"
            )
        
        with col2:
            annual_inc = st.number_input(
                "Annual Income (₹)",
                min_value=300000,
                max_value=25000000,
                value=540000,
                step=10000,
                help="The borrower's annual income (typical range: ₹3,00,000 - ₹2,50,00,000)"
            )
            
            verification_status = st.selectbox(
                "Income Verification Status",
                options=['Not Verified', 'Source Verified', 'Verified'],
                index=1,
                help="Whether the borrower's income has been verified"
            )
            
            initial_list_status = st.selectbox(
                "Initial Listing Status",
                options=[1, 0],
                format_func=lambda x: "Whole" if x == 1 else "Fractional",
                index=0,
                help="Initial listing status of the loan"
            )
    
    with tab2:
        st.markdown('''
            <div style="margin-bottom: 20px;">
                <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#1f77b4" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display: inline-block; vertical-align: middle; margin-right: 10px;">
                    <rect x="1" y="4" width="22" height="16" rx="2" ry="2"/>
                    <line x1="1" y1="10" x2="23" y2="10"/>
                </svg>
                <h3 style="display: inline-block; vertical-align: middle; margin: 0;">Credit History & Profile</h3>
            </div>
        ''', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Credit Score**")
            fico_score = st.slider(
                "FICO Score",
                min_value=300,
                max_value=850,
                value=720,
                step=1,
                help="The borrower's FICO credit score (300-850)"
            )
            
            grade = st.selectbox(
                "Loan Grade",
                options=[0, 1, 2, 3, 4, 5, 6],
                format_func=lambda x: chr(65 + x),  # Convert 0->A, 1->B, etc.
                index=2,
                help="LC assigned loan grade (A=0, B=1, C=2, etc.)"
            )
        
        with col2:
            st.markdown("**Debt & Utilization**")
            dti = st.number_input(
                "Debt-to-Income Ratio (%)",
                min_value=0.0,
                max_value=50.0,
                value=18.0,
                step=0.5,
                help="The borrower's debt-to-income ratio (typical range: 0-50%)"
            )
            
            revol_util = st.number_input(
                "Revolving Line Utilization (%)",
                min_value=0.0,
                max_value=100.0,
                value=45.0,
                step=1.0,
                help="Amount of credit used relative to all available revolving credit"
            )
            
            revol_bal = st.number_input(
                "Revolving Balance (₹)",
                min_value=0,
                max_value=1000000,
                value=100000,
                step=5000,
                help="Total credit revolving balance (typical range: ₹0 - ₹10,00,000)"
            )
        
        with col3:
            st.markdown("**Credit History**")
            delinq_2yrs = st.number_input(
                "Delinquencies (Last 2 Years)",
                min_value=0,
                max_value=15,
                value=0,
                step=1,
                help="Number of 30+ days past-due incidences in the past 2 years"
            )
            
            inq_last_6mths = st.number_input(
                "Credit Inquiries (Last 6 Months)",
                min_value=0,
                max_value=10,
                value=1,
                step=1,
                help="Number of credit inquiries in the last 6 months"
            )
            
            pub_rec = st.number_input(
                "Public Records",
                min_value=0,
                max_value=10,
                value=0,
                step=1,
                help="Number of derogatory public records"
            )
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Account Information**")
            open_acc = st.number_input(
                "Open Credit Lines",
                min_value=0,
                max_value=50,
                value=11,
                step=1,
                help="Number of open credit lines"
            )
            
            total_acc = st.number_input(
                "Total Credit Lines",
                min_value=0,
                max_value=100,
                value=25,
                step=1,
                help="Total number of credit lines"
            )
        
        with col2:
            st.markdown("**Recent Activity**")
            open_rv_12m = st.number_input(
                "Revolving Accounts (Last 12 Months)",
                min_value=0,
                max_value=20,
                value=1,
                step=1,
                help="Number of revolving accounts opened in the last 12 months"
            )
            
            open_rv_24m = st.number_input(
                "Revolving Accounts (Last 24 Months)",
                min_value=0,
                max_value=30,
                value=3,
                step=1,
                help="Number of revolving accounts opened in the last 24 months"
            )
    
    with tab3:
        st.markdown('''
            <div style="margin-bottom: 20px;">
                <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#1f77b4" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display: inline-block; vertical-align: middle; margin-right: 10px;">
                    <line x1="12" y1="20" x2="12" y2="10"/>
                    <line x1="18" y1="20" x2="18" y2="4"/>
                    <line x1="6" y1="20" x2="6" y2="16"/>
                </svg>
                <h3 style="display: inline-block; vertical-align: middle; margin: 0;">Interest Rate Prediction & Analysis</h3>
            </div>
        ''', unsafe_allow_html=True)
        
        # Collect all user inputs
        user_inputs = {
            'loan_amnt': loan_amnt,
            'term': 0 if term == 36 else 1,  # Normalize term
            'grade': grade,
            'annual_inc': annual_inc,
            'dti': dti,
            'delinq_2yrs': delinq_2yrs,
            'inq_last_6mths': inq_last_6mths,
            'open_acc': open_acc,
            'pub_rec': pub_rec,
            'revol_bal': revol_bal,
            'revol_util': revol_util,
            'total_acc': total_acc,
            'initial_list_status': initial_list_status,
            'open_rv_12m': open_rv_12m,
            'open_rv_24m': open_rv_24m,
            'fico_score': fico_score,
            'verification_status': verification_status,
            'purpose': purpose
        }
        
        # Create prediction button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            predict_button = st.button(
                "Predict Interest Rate", 
                type="primary", 
                use_container_width=True
            )
        
    
        if predict_button:
            with st.spinner("Calculating your personalized interest rate..."):
                # Create feature vector
                features_df = create_input_features(user_inputs)
                
                # Make prediction
                model = models[selected_model]
                prediction = model.predict(features_df)[0]
                scaler = load_scaler()
                features_df['int_rate'] = prediction
                original_data = scaler.inverse_transform(features_df[numeric_features])
                int_rate_value = original_data[0, numeric_features.index("int_rate")]
                prediction = int_rate_value

                # Display prediction
                st.markdown("---")
                st.markdown("### Predicted Interest Rate")
                st.markdown(f'<p class="prediction-value">{prediction:.2f}%</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Display additional insights
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    monthly_payment = (loan_amnt * (prediction/100/12) * (1 + prediction/100/12)**(term)) / ((1 + prediction/100/12)**(term) - 1)
                    st.metric(
                        "Monthly Payment (Approx.)",
                        f"₹{monthly_payment:,.2f}"
                    )
                
                with col2:
                    total_interest = (loan_amnt * (prediction/100) * (term/12))
                    st.metric(
                        "Total Interest",
                        f"₹{total_interest:,.2f}"
                    )
                
                with col3:
                    total_repayment = loan_amnt + total_interest
                    st.metric(
                        "Total Repayment",
                        f"₹{total_repayment:,.2f}"
                    )
                
                # Risk assessment
                st.markdown("---")
                st.subheader("Profile Assessment")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Credit Strength Factors:**")
                    if fico_score >= 740:
                        st.success("✓ Excellent FICO score")
                    elif fico_score >= 670:
                        st.info("→ Good FICO score")
                    else:
                        st.warning("⚠ Fair FICO score")
                    
                    if dti < 20:
                        st.success("✓ Low debt-to-income ratio")
                    elif dti < 35:
                        st.info("→ Moderate debt-to-income ratio")
                    else:
                        st.warning("⚠ High debt-to-income ratio")
                    
                    if delinq_2yrs == 0:
                        st.success("✓ No recent delinquencies")
                    else:
                        st.warning(f"⚠ {delinq_2yrs} delinquencies in last 2 years")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown("**Loan Characteristics:**")
                    st.write(f"• Loan Grade: **{chr(65 + grade)}**")
                    st.write(f"• Income/Loan Ratio: **{user_inputs['annual_inc']/loan_amnt:.2f}x**")
                    st.write(f"• Verification: **{verification_status}**")
                    st.write(f"• Term: **{term} months**")
                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style"padding: 30px; border-radius: 10px; border-left: 5px solid #1f77b4; margin: 20px 0;">
                    <h4 style="color: #1f77b4; margin-top: 0;">
                        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-right: 10px;">
                            <circle cx="12" cy="12" r="10"/>
                            <line x1="12" y1="16" x2="12" y2="12"/>
                            <line x1="12" y1="8" x2="12.01" y2="8"/>
                        </svg>
                        Ready to Predict
                    </h4>
                    <p style="margin-bottom: 10px;">Review your inputs in the <strong>Loan Details</strong> and <strong>Credit Profile</strong> tabs, then click the <strong>Predict Interest Rate</strong> button above to:</p>
                    <ul style="margin-left: 20px;">
                        <li>Get your personalized interest rate prediction</li>
                        <li>Calculate estimated monthly payments</li>
                        <li>Receive a comprehensive credit profile assessment</li>
                        <li>Compare predictions across all available models</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 20px;'>
            <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#1f77b4" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-bottom: 10px;">
                <rect x="2" y="5" width="20" height="14" rx="2"/>
                <line x1="2" y1="10" x2="22" y2="10"/>
            </svg>
            <p style="font-size: 1.2rem; font-weight: bold; margin: 10px 0;">AcuRate</p>
            <p style="margin: 5px 0;">Personalized Interest Rate Prediction System</p>
            <p style='font-size: 0.9rem; color: #999;'>Powered by Machine Learning • Built with Streamlit • Trained on Lending Club Dataset</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
