import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image

# Load models and data
models = [joblib.load('XGB.pkl')]
dummy_columns = joblib.load('dummy_columns.pkl')

# Set up page config
st.set_page_config(
    page_title="👩‍💼 Employee Churn Prediction",
    page_icon=":office_worker:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with softer colors
st.markdown("""
    <style>
    :root {
        --primary: #5a7faa;
        --secondary: #2a7f9d;
        --accent: #6ec1e8;
        --background: #f8f9fa;
        --card: #ffffff;
        --text: #333333;
        --positive: #5cb85c;  /* Softer green */
        --negative: #d9534f;  /* Softer red */
    }
    
    .main {
        background-color: var(--background);
    }
    
    /* Transparent tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: transparent;
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        padding: 10px 20px;
        border-radius: 8px 8px 0 0;
        transition: all 0.3s;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(90, 127, 170, 0.1);
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--card) !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .header-container {
        display: flex;
        align-items: center;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        padding: 2rem;
        border-radius: 10px;
        color: white;
    }
    
    .header-text {
        padding-left: 2rem;
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .header-subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    .prediction-card {
        border-radius: 12px;
        padding: 22px;
        margin: 15px 0;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        background-color: var(--card);
        border: 1px solid rgba(0,0,0,0.05);
    }
    
    .positive {
        border-left: 5px solid var(--negative);
        background-color: rgba(217, 83, 79, 0.03);
    }
    
    .negative {
        border-left: 5px solid var(--positive);
        background-color: rgba(92, 184, 92, 0.03);
    }
    
    .sidebar .sidebar-content {
        background-color: var(--card);
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .stSlider>div>div>div>div {
        background: var(--accent) !important;
    }
    
    .stButton>button {
        background-color: var(--secondary);
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background-color: var(--primary);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .footer {
        text-align: center;
        padding: 1rem;
        margin-top: 2rem;
        color: var(--text);
        font-size: 0.9rem;
    }
    
    .team-credits {
        background-color: rgba(74, 111, 165, 0.1);
        padding: 15px;
        border-radius: 8px;
        margin: 20px 0;
    }
    
    .team-members {
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
        gap: 8px 15px;
        margin: 10px 0;
    }
    
    /* Tab content background transparent */
    .stTabs [role="tabpanel"] {
        background-color: transparent !important;
        padding: 0 !important;
    }
    
    /* Remove white background from main content */
    .block-container {
        padding-top: 2rem;
        background-color: transparent !important;
    }
    
    /* Make tab content blend with page background */
    div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column"] {
        background-color: transparent !important;
    }
    </style>
    """, unsafe_allow_html=True)

# App header with large image
# App header with large image and team credits
def render_header():
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(Image.open("churn.png"), width=400)
    with col2:
        st.markdown("""
        <div class="header-text">
            <h1 class="header-title">👩‍💼Employee Churn Prediction</h1>
            <p class="header-subtitle">
                Predict churn risks • Improve retention • Strengthen your workforce
            </p>
        </div>
        """, unsafe_allow_html=True)

render_header()

# Create tabs with transparent background
tab1, tab2 = st.tabs(["📊 Prediction Dashboard", "📈 Analytics Insights"])

with tab1:
    # Sidebar for input
    with st.sidebar:
        st.markdown("## Employee Details")
        st.markdown("Complete the form to assess retention risk")
        
        with st.expander("Personal Factors", expanded=True):
            satisfaction_level = st.slider("Satisfaction Level", 0.0, 1.0, 0.5, 0.01,
                                         help="Employee's overall job satisfaction")
            st.caption(f"Current value: {satisfaction_level:.0%}")
            
            department = st.selectbox("Department", 
                                   ['sales', 'technical', 'support', 'IT', 'hr', 
                                    'accounting', 'marketing', 'product_mng', 
                                    'management', 'RandD'],
                                   help="Employee's department")
            
            salary = st.selectbox("Salary Level", ['low', 'medium', 'high'],
                                 help="Employee's salary tier")
        
        with st.expander("Performance Metrics", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                last_evaluation = st.slider("Evaluation Score", 0.0, 1.0, 0.7, 0.05)
            with col2:
                number_project = st.number_input("Project Count", 1, 10, 3)
            
            average_montly_hours = st.slider("Monthly Hours", 50, 400, 160, 10,
                                           help="Average working hours per month")
        
        with st.expander("Employment History", expanded=False):
            time_spend_company = st.select_slider("Company Tenure", 
                                                options=list(range(1, 21)), 
                                                value=3)
            
            work_accident = st.radio("Work Accident", ["No", "Yes"], index=0,
                                   horizontal=True)
            
            promotion_last_5years = st.radio("Recent Promotion", ["No", "Yes"], 
                                           index=0, horizontal=True)
        
        if st.button("Assess Retention Risk", type="primary"):
            predict_clicked = True
        else:
            predict_clicked = False

    # Main content area
    if predict_clicked:
        # Convert inputs
        work_accident = 1 if work_accident == "Yes" else 0
        promotion_last_5years = 1 if promotion_last_5years == "Yes" else 0

        # Create input DataFrame
        input_df = pd.DataFrame([{
            'satisfaction_level': satisfaction_level,
            'last_evaluation': last_evaluation,
            'number_project': number_project,
            'average_montly_hours': average_montly_hours,
            'time_spend_company': time_spend_company,
            'Work_accident': work_accident,
            'promotion_last_5years': promotion_last_5years,
            'Departments': department,
            'salary': salary,
        }])

        # Encode categorical variables
        input_df = pd.get_dummies(input_df, columns=['Departments', 'salary'], drop_first=True)

        # Ensure same columns as training set
        for col in dummy_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[dummy_columns]  # Reorder columns

        # Predict with all models
        predictions = [model.predict(input_df)[0] for model in models]
        prediction_probs = [model.predict_proba(input_df)[0][1] for model in models]
        final_prediction = max(set(predictions), key=predictions.count)
        avg_prob = np.mean(prediction_probs)

        # Show result
        st.subheader("Retention Risk Assessment")
        
        if final_prediction == 1:
            risk_level = "High Risk"
            risk_color = "var(--negative)"
            card_class = "positive"
            message = "⚠️ Higher probability of employee leaving"
            icon = "⚠️"
        else:
            risk_level = "Low Risk"
            risk_color = "var(--positive)"
            card_class = "negative"
            message = "✅ Employee likely to stay"
            icon = "✅"
        
        # Risk card with softer styling
        st.markdown(f"""
        <div class="prediction-card {card_class}">
            <div style="display: flex; align-items: center; margin-bottom: 12px;">
                <span style="font-size: 1.8rem; margin-right: 12px; opacity: 0.9;">{icon}</span>
                <div>
                    <h2 style="color:{risk_color}; margin:0; font-weight:600; opacity: 0.9;">{risk_level}</h2>
                    <p style="margin:0; font-size: 1rem; opacity: 0.8;">{message}</p>
                </div>
            </div>
            <div style="background: rgba(0,0,0,0.03); padding: 12px; border-radius: 8px;">
                <h4 style="margin-top:0; font-weight:500; opacity: 0.9;">Risk Probability: <strong>{avg_prob*100:.1f}%</strong></h4>
                <div style="height: 6px; background: #f0f0f0; border-radius: 3px; margin: 8px 0;">
                    <div style="height: 100%; width: {avg_prob*100}%; background: {risk_color}; border-radius: 3px; opacity: 0.8;"></div>
                </div>
                <p style="font-size: 0.85rem; margin-bottom:0; opacity: 0.7;">Scale: 0% (No risk) → 100% (Certain to leave)</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    # Analytics Insights Tab
    st.header("Retention Overview")
    st.write("""
    ## Key Retention Metrics
    Understand your workforce stability at a glance
    """)
    
    # Simple metrics cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Overall Retention Rate", "82%", "+2% from last quarter")
    with col2:
        st.metric("Average Satisfaction", "6.8/10", "-0.3 from last quarter")
    with col3:
        st.metric("Avg Company Tenure", "3.2 years", "stable")
    
    st.markdown("---")
    
    # Simple charts section
    st.subheader("Top Retention Factors")
    
    # Factor importance chart
    factors = pd.DataFrame({
        'Factor': ['Satisfaction', 'Workload', 'Salary', 'Career Growth', 'Work Environment'],
        'Impact': [85, 72, 68, 63, 58]
    })
    st.bar_chart(factors.set_index('Factor'))
    
    st.markdown("---")
    
    # Simple recommendations
    st.subheader("Quick Recommendations")
    st.write("""
    - **Focus on satisfaction**: Employees with satisfaction below 5/10 are 3x more likely to leave
    - **Monitor workload**: Those working >200 hours/month show higher turnover
    - **Review compensation**: Low salary tier employees have 25% higher churn
    - **Career development**: Employees without promotion in 3+ years are at risk
    """)
    
    # Department comparison
    st.markdown("---")
    st.subheader("By Department")
    dept_data = pd.DataFrame({
        'Department': ['Sales', 'Engineering', 'technical', 'HR', 'Management'],
        'Retention Rate': [75, 88, 82, 91, 89],
        'Avg Satisfaction': [6.2, 7.1, 6.8, 7.4, 7.3]
    })
    st.dataframe(dept_data.style.highlight_max(axis=0, color='red'))

# Footer with team credits
# Footer with enhanced team credits
# Footer with enhanced team credits
# Enhanced Footer with Team Credits
st.markdown("""
<div class="footer">
    <div class="team-credits">
        <p style="font-weight: 600; text-align: center; margin-bottom: 10px;">Project Development Team</p>
        <div class="team-members">
            <span>• Ahmed Mohamed</span>
            <span>• Theodore Naguib</span>
            <span>• Malak Torky</span>
            <span>• Shrouk Emam</span>
            <span>• Salah Eldin Mohamed</span>
            <span>• Seif Ahmed</span>
        </div>
        <p style="text-align: center; margin-top: 10px;">
            Supervised by: <span style="font-weight: 600;">Eng. Mahmoud Talaat</span>
        </p>
    </div>
    <p style="margin-top: 20px;">Employee Retention Pro • Powered by HR Analytics • v2.1</p>
</div>
""", unsafe_allow_html=True)
