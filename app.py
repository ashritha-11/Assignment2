import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings

warnings.filterwarnings("ignore")

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Loan Risk Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_files():

    model = joblib.load("loan_model.pkl")

    scaler = joblib.load("scaler.pkl")

    return model, scaler

model, scaler = load_files()

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
    background-color: #0B1120;
    color: white;
}

/* Main Container */
.block-container {
    padding-top: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #111827;
    border-right: 1px solid #1F2937;
    width: 320px !important;
}

/* Sidebar Text */
section[data-testid="stSidebar"] * {
    color: white !important;
}

/* Hide Streamlit Branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Title */
.main-title {
    font-size: 48px;
    font-weight: 700;
    color: white;
    margin-bottom: 5px;
}

/* Subtitle */
.sub-title {
    color: #94A3B8;
    font-size: 18px;
    margin-bottom: 30px;
}

/* Cards */
.metric-card {
    background: linear-gradient(145deg, #111827, #1E293B);
    padding: 30px;
    border-radius: 20px;
    border: 1px solid #334155;
    text-align: center;
    box-shadow: 0px 10px 25px rgba(0,0,0,0.35);
}

/* Metric Values */
.metric-value {
    font-size: 38px;
    font-weight: 700;
    color: white;
}

/* Metric Labels */
.metric-label {
    font-size: 16px;
    color: #CBD5E1;
    margin-top: 8px;
}

/* Result Card */
.result-card {
    background: linear-gradient(145deg, #111827, #1E293B);
    padding: 35px;
    border-radius: 20px;
    border: 1px solid #334155;
    box-shadow: 0px 10px 25px rgba(0,0,0,0.35);
}

/* Button */
.stButton > button {
    width: 100%;
    height: 58px;
    border: none;
    border-radius: 16px;
    background: linear-gradient(to right, #0066ff, #00ccff);
    color: white;
    font-size: 18px;
    font-weight: 600;
    transition: 0.3s;
}

.stButton > button:hover {
    transform: scale(1.02);
    background: linear-gradient(to right, #00ccff, #0066ff);
}

/* Progress Bar */
.stProgress > div > div > div {
    background-color: #00ccff;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="main-title">
🏦 AI Loan Risk Predictor
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="sub-title">
AI-Powered Loan Approval & Risk Analysis System
</div>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("Applicant Information")

    age = st.slider(
        "Age",
        18,
        70,
        30
    )

    income = st.number_input(
        "Annual Income",
        min_value=10000,
        max_value=1000000,
        value=50000
    )

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=1000,
        max_value=500000,
        value=100000
    )

    credit_score = st.slider(
        "Credit Score",
        300,
        900,
        650
    )

    employment_years = st.slider(
        "Employment Years",
        0,
        40,
        5
    )

# =========================================================
# DISPLAY CARDS
# =========================================================

c1, c2, c3 = st.columns(3)

with c1:

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">₹ {income}</div>
        <div class="metric-label">Annual Income</div>
    </div>
    """, unsafe_allow_html=True)

with c2:

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{credit_score}</div>
        <div class="metric-label">Credit Score</div>
    </div>
    """, unsafe_allow_html=True)

with c3:

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{employment_years} Years</div>
        <div class="metric-label">Employment Experience</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =========================================================
# INPUT DATAFRAME
# =========================================================

input_data = pd.DataFrame([{

    "Age": age,
    "Income": income,
    "LoanAmount": loan_amount,
    "CreditScore": credit_score,
    "EmploymentYears": employment_years

}])

# =========================================================
# PREDICTION
# =========================================================

if st.button("Predict Loan Approval"):

    try:

        # Scale Input
        scaled_data = scaler.transform(input_data)

        # Prediction
        prediction = model.predict(scaled_data)[0]

        probability = model.predict_proba(
            scaled_data
        )[0][1] * 100

        # =================================================
        # RISK LEVEL
        # =================================================

        if probability >= 75:

            risk = "🟢 LOW RISK"

            decision = "✅ Loan Approved"

        elif probability >= 50:

            risk = "🟠 MEDIUM RISK"

            decision = "⚠️ Review Required"

        else:

            risk = "🔴 HIGH RISK"

            decision = "❌ Loan Rejected"

        # =================================================
        # RESULTS
        # =================================================

        st.markdown("## Prediction Result")

        r1, r2 = st.columns(2)

        with r1:

            st.markdown(f"""
            <div class="result-card">
                <h1>{probability:.2f}%</h1>
                <p>Approval Probability</p>
            </div>
            """, unsafe_allow_html=True)

            st.progress(int(probability))

        with r2:

            st.markdown(f"""
            <div class="result-card">
                <h2>{risk}</h2>
                <p>{decision}</p>
            </div>
            """, unsafe_allow_html=True)

        # =================================================
        # CUSTOMER SEGMENT
        # =================================================

        st.markdown("## Customer Profile")

        if income > 100000 and credit_score > 700:

            st.success("Premium Customer")

        elif credit_score < 500:

            st.error("High Risk Customer")

        else:

            st.info("Standard Customer")

    except Exception as e:

        st.error(f"Prediction Error: {e}")
