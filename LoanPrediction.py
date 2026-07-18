import numpy as np
import pandas as pd
import  streamlit as st
import joblib
import os
import gdown

finall_pipe=joblib.load('LoanPrediction_Model.pkl')


st.set_page_config(page_title="Loan Approval AI", layout="wide", page_icon="🏦")

# Auto download model
MODEL_PATH = "loan_Model.pkl"
GDRIVE_URL = "https://drive.google.com/uc?id=1EP-hoWBXuO2EExpF5aCIBc3qZYtic15H"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model... Please wait 20-30 seconds"):
        gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
    st.success("Model Downloaded!")

model = joblib.load(MODEL_PATH)

# Header
st.title("🏦 AI Loan Approval Predictor")
st.markdown("Fill the applicant details below and get instant loan approval prediction")
st.divider()

# Create 3 columns for clean UI
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Personal Info")
    person_gender = st.selectbox("Gender", ["male", "female"])
    person_age = st.slider("Age", 20, 80, 30)
    person_education = st.selectbox("Education", ["High School", "Bachelor", "Master", "Doctorate"])
    person_home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
    person_emp_exp = st.slider("Employment Experience in Years", 0, 40, 5)

with col2:
    st.subheader("💰 Financial Info")
    person_income = st.number_input("Annual Income $", min_value=0, value=50000, step=1000)
    loan_amnt = st.number_input("Loan Amount Requested $", min_value=1000, value=10000, step=500)
    loan_int_rate = st.slider("Interest Rate %", 1.0, 30.0, 10.5, 0.1)
    loan_percent_income = st.slider("Loan % of Income", 0.0, 1.0, 0.2, 0.01)

with col3:
    st.subheader("📊 Credit Info")
    loan_intent = st.selectbox("Loan Intent", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
    cb_person_cred_hist_length = st.slider("Credit History Length in Years", 0, 30, 5)
    credit_score = st.slider("Credit Score", 300, 850, 650)
    previous_loan_defaults_on_file = st.selectbox("Previous Loan Default", ["Yes", "No"])

# Map to match training encoding
def encode_inputs():
    gender_map = {"male": 1, "female": 0}
    education_map = {"High School": 0, "Bachelor": 1, "Master": 2, "Doctorate": 3}
    home_map = {"RENT": 0, "OWN": 1, "MORTGAGE": 2, "OTHER": 3}
    intent_map = {"PERSONAL": 0, "EDUCATION": 1, "MEDICAL": 2, "VENTURE": 3, "HOMEIMPROVEMENT": 4, "DEBTCONSOLIDATION": 5}
    default_map = {"Yes": 1, "No": 0}

    data = {
        'person_age': person_age,
        'person_gender': gender_map[person_gender],
        'person_education': education_map[person_education],
        'person_income': person_income,
        'person_emp_exp': person_emp_exp,
        'person_home_ownership': home_map[person_home_ownership],
        'loan_amnt': loan_amnt,
        'loan_intent': intent_map[loan_intent],
        'loan_int_rate': loan_int_rate,
        'loan_percent_income': loan_percent_income,
        'cb_person_cred_hist_length': cb_person_cred_hist_length,
        'credit_score': credit_score,
        'previous_loan_defaults_on_file': default_map[previous_loan_defaults_on_file]
    }
    return pd.DataFrame([data])

st.divider()
predict_btn = st.button("🚀 Predict Loan Status", type="primary", use_container_width=True)

if predict_btn:
    input_df = encode_inputs()

    with st.spinner("Analyzing..."):
        prediction = model.predict(input_df)
        proba = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None

    st.subheader("Result")
    if prediction[0] == 1:
        st.success(f"✅ Loan Approved")
        st.balloons()
    else:
        st.error(f"❌ Loan Not Approved")

    if proba is not None:
        st.metric("Approval Probability", f"{proba*100:.2f}%")

    with st.expander("See Input Data"):
        st.dataframe(input_df, use_container_width=True)

st.caption("Note: This model is for demo purposes only")