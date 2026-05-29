
import streamlit as st
import pandas as pd
import pickle
import numpy as np

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="AI Loan Risk Predictor",
    page_icon="🏦",
    layout="wide"
)

# ==========================================
# LOAD MODEL
# ==========================================

model = pickle.load(open("loan_model.pkl", "rb"))

scaler = pickle.load(open("scaler.pkl", "rb"))

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""

<style>

.main {
    background-color: #0E1117;
}

.stButton>button {
    width: 100%;
    background: linear-gradient(to right, #0066ff, #00ccff);
    color: white;
    border-radius: 10px;
    height: 3em;
    font-size: 18px;
    font-weight: bold;
    border: none;
}

</style>

""", unsafe_allow_html=True)

# ==========================================
# TITLE
# ==========================================

st.title("🏦 AI-Powered Loan Approval Prediction System")

st.markdown("""
Predict whether a customer loan should be approved using Machine Learning.
""")

st.markdown("---")

# ==========================================
# SIDEBAR INPUTS
# ==========================================

st.sidebar.header("Applicant Information")

age = st.sidebar.slider(
    "Age",
    18,
    70,
    30
)

income = st.sidebar.number_input(
    "Annual Income",
    10000,
    1000000,
    50000
)

loan_amount = st.sidebar.number_input(
    "Loan Amount",
    1000,
    500000,
    100000
)

credit_score = st.sidebar.slider(
    "Credit Score",
    300,
    900,
    650
)

employment_years = st.sidebar.slider(
    "Employment Years",
    0,
    40,
    5
)

# ==========================================
# INPUT DATAFRAME
# ==========================================

input_data = pd.DataFrame([{

    "Age": age,
    "Income": income,
    "LoanAmount": loan_amount,
    "CreditScore": credit_score,
    "EmploymentYears": employment_years

}])

# ==========================================
# PREDICTION BUTTON
# ==========================================

if st.button("Predict Loan Approval"):

    try:

        scaled_data = scaler.transform(input_data)

        prediction = model.predict(scaled_data)[0]

        probability = model.predict_proba(
            scaled_data
        )[0][1] * 100

        st.markdown("## 📊 Prediction Result")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                label="Approval Probability",
                value=f"{probability:.2f}%"
            )

            st.progress(int(probability))

        with col2:

            if prediction == 1:
                st.success("✅ Loan Approved")
            else:
                st.error("❌ Loan Rejected")

        # ======================================
        # RISK LEVEL
        # ======================================

        if probability >= 75:
            risk = "🟢 LOW RISK"

        elif probability >= 50:
            risk = "🟠 MEDIUM RISK"

        else:
            risk = "🔴 HIGH RISK"

        st.info(f"Risk Level: {risk}")

        # ======================================
        # CUSTOMER SEGMENT
        # ======================================

        st.markdown("## 👤 Customer Analysis")

        if income > 100000 and credit_score > 700:
            segment = "Premium Customer"

        elif credit_score < 500:
            segment = "Risky Customer"

        else:
            segment = "Standard Customer"

        st.success(f"Customer Segment: {segment}")

    except Exception as e:

        st.error(f"Prediction Error: {e}")

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.markdown("""
### Technologies Used

- Logistic Regression
- Random Forest
- PCA
- K-Means
- Streamlit
- Scikit-learn
""")

