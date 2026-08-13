

import streamlit as st
import pandas as pd
import numpy as np
import pickle


# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="Credit Card Default Prediction",
    page_icon="💳",
    layout="wide"
)


# ==========================================
# Load Trained Model
# ==========================================

@st.cache_resource
def load_model():

    with open("Models/model.pkl", "rb") as file:
        model = pickle.load(file)

    return model


model = load_model()


# ==========================================
# Page Title
# ==========================================

st.title("💳 Credit Card Default Risk Prediction")

st.write(
    "Enter the customer's financial and repayment information "
    "to predict the risk of defaulting on the next payment."
)

st.markdown("---")


# ==========================================
# Customer Information
# ==========================================

st.header("👤 Customer Information")

col1, col2, col3 = st.columns(3)


with col1:

    limit_bal = st.number_input(
        "Credit Limit",
        min_value=0.0,
        value=50000.0,
        step=5000.0
    )


with col2:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )


with col3:

    sex = st.selectbox(
        "Sex",
        options=[1, 2],
        format_func=lambda x:
            "Male" if x == 1 else "Female"
    )


col1, col2 = st.columns(2)


with col1:

    education = st.selectbox(
        "Education",
        options=[1, 2, 3, 4],
        format_func=lambda x: {
            1: "Graduate School",
            2: "University",
            3: "High School",
            4: "Other"
        }[x]
    )


with col2:

    marriage = st.selectbox(
        "Marriage",
        options=[1, 2, 3],
        format_func=lambda x: {
            1: "Married",
            2: "Single",
            3: "Other"
        }[x]
    )


# ==========================================
# Repayment History
# ==========================================

st.markdown("---")

st.header("📅 Repayment History")

st.info(
    """
Repayment status represents the customer's payment history.
Higher values generally indicate greater payment delays.
"""
)

col1, col2, col3 = st.columns(3)


with col1:

    pay_0 = st.number_input(
        "PAY_0 - Most Recent",
        min_value=-2,
        max_value=8,
        value=0
    )


with col2:

    pay_2 = st.number_input(
        "PAY_2 - 2 Months Ago",
        min_value=-2,
        max_value=8,
        value=0
    )


with col3:

    pay_3 = st.number_input(
        "PAY_3 - 3 Months Ago",
        min_value=-2,
        max_value=8,
        value=0
    )


col1, col2, col3 = st.columns(3)


with col1:

    pay_4 = st.number_input(
        "PAY_4 - 4 Months Ago",
        min_value=-2,
        max_value=8,
        value=0
    )


with col2:

    pay_5 = st.number_input(
        "PAY_5 - 5 Months Ago",
        min_value=-2,
        max_value=8,
        value=0
    )


with col3:

    pay_6 = st.number_input(
        "PAY_6 - 6 Months Ago",
        min_value=-2,
        max_value=8,
        value=0
    )


# ==========================================
# Billing Amounts
# ==========================================

st.markdown("---")

st.header("💰 Six-Month Billing History")

col1, col2, col3 = st.columns(3)


with col1:

    bill_amt1 = st.number_input(
        "Bill Amount 1",
        value=10000.0
    )

    bill_amt2 = st.number_input(
        "Bill Amount 2",
        value=10000.0
    )


with col2:

    bill_amt3 = st.number_input(
        "Bill Amount 3",
        value=10000.0
    )

    bill_amt4 = st.number_input(
        "Bill Amount 4",
        value=10000.0
    )


with col3:

    bill_amt5 = st.number_input(
        "Bill Amount 5",
        value=10000.0
    )

    bill_amt6 = st.number_input(
        "Bill Amount 6",
        value=10000.0
    )


# ==========================================
# Payment Amounts
# ==========================================

st.markdown("---")

st.header("💵 Six-Month Payment History")

col1, col2, col3 = st.columns(3)


with col1:

    pay_amt1 = st.number_input(
        "Payment Amount 1",
        min_value=0.0,
        value=2000.0
    )

    pay_amt2 = st.number_input(
        "Payment Amount 2",
        min_value=0.0,
        value=2000.0
    )


with col2:

    pay_amt3 = st.number_input(
        "Payment Amount 3",
        min_value=0.0,
        value=2000.0
    )

    pay_amt4 = st.number_input(
        "Payment Amount 4",
        min_value=0.0,
        value=2000.0
    )


with col3:

    pay_amt5 = st.number_input(
        "Payment Amount 5",
        min_value=0.0,
        value=2000.0
    )

    pay_amt6 = st.number_input(
        "Payment Amount 6",
        min_value=0.0,
        value=2000.0
    )


# ==========================================
# Prediction Button
# ==========================================

st.markdown("---")

predict_button = st.button(
    "🔍 Predict Default Risk",
    use_container_width=True
)


# ==========================================
# Prediction
# ==========================================

if predict_button:

    # --------------------------------------
    # Feature Engineering
    # --------------------------------------

    avg_bill_amt = np.mean([
        bill_amt1,
        bill_amt2,
        bill_amt3,
        bill_amt4,
        bill_amt5,
        bill_amt6
    ])

    avg_pay_amt = np.mean([
        pay_amt1,
        pay_amt2,
        pay_amt3,
        pay_amt4,
        pay_amt5,
        pay_amt6
    ])

    # Avoid division by zero
    if avg_bill_amt > 0:

        pay_to_bill_ratio = avg_pay_amt / avg_bill_amt

    else:

        pay_to_bill_ratio = 0

    max_delay = max([
        pay_0,
        pay_2,
        pay_3,
        pay_4,
        pay_5,
        pay_6
    ])


    # --------------------------------------
    # Create Input DataFrame
    # --------------------------------------

    input_data = pd.DataFrame({

        "LIMIT_BAL": [limit_bal],

        "SEX": [sex],

        "EDUCATION": [education],

        "MARRIAGE": [marriage],

        "AGE": [age],

        "PAY_0": [pay_0],

        "PAY_2": [pay_2],

        "PAY_3": [pay_3],

        "PAY_4": [pay_4],

        "PAY_5": [pay_5],

        "PAY_6": [pay_6],

        "BILL_AMT1": [bill_amt1],

        "BILL_AMT2": [bill_amt2],

        "BILL_AMT3": [bill_amt3],

        "BILL_AMT4": [bill_amt4],

        "BILL_AMT5": [bill_amt5],

        "BILL_AMT6": [bill_amt6],

        "PAY_AMT1": [pay_amt1],

        "PAY_AMT2": [pay_amt2],

        "PAY_AMT3": [pay_amt3],

        "PAY_AMT4": [pay_amt4],

        "PAY_AMT5": [pay_amt5],

        "PAY_AMT6": [pay_amt6],

        "avg_bill_amt": [avg_bill_amt],

        "avg_pay_amt": [avg_pay_amt],

        "pay_to_bill_ratio": [pay_to_bill_ratio],

        "max_delay": [max_delay]

    })


    # --------------------------------------
    # Prediction
    # --------------------------------------

    prediction = model.predict(input_data)[0]


    # --------------------------------------
    # Probability
    # --------------------------------------

    if hasattr(model, "predict_proba"):

        probability = model.predict_proba(
            input_data
        )[0][1]

    else:

        probability = None


    # ======================================
    # Result
    # ======================================

    st.markdown("---")

    st.header("📊 Prediction Result")


    if prediction == 1:

        st.error("🔴 High Risk of Default")

        risk_level = "HIGH"

    else:

        st.success("🟢 Low Risk of Default")

        risk_level = "LOW"


    # ======================================
    # Probability
    # ======================================

    if probability is not None:

        default_probability = probability * 100

        st.metric(
            "Default Probability",
            f"{default_probability:.2f}%"
        )

        st.progress(
            float(probability)
        )


    # ======================================
    # Risk Level
    # ======================================

    st.metric(
        "Risk Level",
        risk_level
    )


    # ======================================
    # Simple Risk Indicators
    # ======================================

    st.subheader("⚠ Risk Indicators")

    indicators = []


    # Recent repayment delay

    if pay_0 > 0:

        indicators.append(
            "Recent repayment delay detected."
        )


    # Maximum delay

    if max_delay >= 2:

        indicators.append(
            "Significant repayment delay found in recent history."
        )


    # Payment to bill ratio

    if pay_to_bill_ratio < 0.20:

        indicators.append(
            "Payments are low compared with the average bill amount."
        )


    # Credit limit

    if limit_bal < 30000:

        indicators.append(
            "Credit limit is relatively low."
        )


    if len(indicators) == 0:

        st.success(
            "No major risk indicators detected from the entered information."
        )

    else:

        for indicator in indicators:

            st.warning(indicator)


# ==========================================
# About Section
# ==========================================

st.markdown("---")

with st.expander("ℹ️ About This Project"):

    st.write(
        """
        This project predicts whether a credit card customer
        is likely to default on their next payment.

        The model uses demographic information, repayment
        history, billing history, and payment history.

        Machine Learning Models used in the project:

        • Logistic Regression
        • Decision Tree
        • Random Forest

        The models are evaluated using Stratified K-Fold
        Cross-Validation.
        """
    )
