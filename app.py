# ==========================================
# Credit Card Default Risk Prediction
# Beginner-Level Streamlit Application
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os


# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="Credit Card Default Prediction",
    page_icon="💳",
    layout="wide"
)


# ==========================================
# File Paths
# ==========================================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "models",
    "model.pkl"
)

FEATURE_COLUMNS_PATH = os.path.join(
    os.path.dirname(__file__),
    "models",
    "feature_columns.pkl"
)


# ==========================================
# Load Model
# ==========================================

@st.cache_resource
def load_model():

    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    return model


# ==========================================
# Load Feature Columns
# ==========================================

@st.cache_resource
def load_feature_columns():

    with open(FEATURE_COLUMNS_PATH, "rb") as file:
        feature_columns = pickle.load(file)

    return feature_columns


# ==========================================
# Load Saved Files
# ==========================================

try:

    model = load_model()

    feature_columns = load_feature_columns()

except FileNotFoundError:

    st.error(
        "Model files were not found. "
        "Please check the models folder."
    )

    st.stop()

except Exception as e:

    st.error("Unable to load the trained model.")

    st.code(str(e))

    st.stop()


# ==========================================
# Title
# ==========================================

st.title("💳 Credit Card Default Risk Prediction")

st.write(
    """
This application predicts whether a credit card customer
is likely to default on their next payment using their
demographic information, repayment history, billing history,
and payment history.
"""
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
PAY values describe the customer's repayment status.
Higher positive values generally indicate payment delays.
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
# Billing History
# ==========================================

st.markdown("---")

st.header("💰 Six-Month Billing History")


col1, col2, col3 = st.columns(3)


with col1:

    bill_amt1 = st.number_input(
        "BILL_AMT1",
        value=10000.0
    )

    bill_amt2 = st.number_input(
        "BILL_AMT2",
        value=10000.0
    )


with col2:

    bill_amt3 = st.number_input(
        "BILL_AMT3",
        value=10000.0
    )

    bill_amt4 = st.number_input(
        "BILL_AMT4",
        value=10000.0
    )


with col3:

    bill_amt5 = st.number_input(
        "BILL_AMT5",
        value=10000.0
    )

    bill_amt6 = st.number_input(
        "BILL_AMT6",
        value=10000.0
    )


# ==========================================
# Payment History
# ==========================================

st.markdown("---")

st.header("💵 Six-Month Payment History")


col1, col2, col3 = st.columns(3)


with col1:

    pay_amt1 = st.number_input(
        "PAY_AMT1",
        min_value=0.0,
        value=2000.0
    )

    pay_amt2 = st.number_input(
        "PAY_AMT2",
        min_value=0.0,
        value=2000.0
    )


with col2:

    pay_amt3 = st.number_input(
        "PAY_AMT3",
        min_value=0.0,
        value=2000.0
    )

    pay_amt4 = st.number_input(
        "PAY_AMT4",
        min_value=0.0,
        value=2000.0
    )


with col3:

    pay_amt5 = st.number_input(
        "PAY_AMT5",
        min_value=0.0,
        value=2000.0
    )

    pay_amt6 = st.number_input(
        "PAY_AMT6",
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

    # ======================================
    # Feature Engineering
    # ======================================

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

        pay_to_bill_ratio = (
            avg_pay_amt / avg_bill_amt
        )

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


    # ======================================
    # Create Raw Customer Data
    # ======================================

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


    # ======================================
    # Handle Categorical Features
    # ======================================

    input_data = pd.get_dummies(
        input_data,
        columns=[
            "SEX",
            "EDUCATION",
            "MARRIAGE"
        ],
        dtype=int
    )


    # ======================================
    # Match Training Columns
    # ======================================

    input_data = input_data.reindex(
        columns=feature_columns,
        fill_value=0
    )


    # ======================================
    # Make Prediction
    # ======================================

    try:

        prediction = model.predict(input_data)[0]

    except Exception as e:

        st.error(
            "The input columns do not match the trained model."
        )

        st.code(str(e))

        st.stop()


    # ======================================
    # Default Probability
    # ======================================

    probability = None

    if hasattr(model, "predict_proba"):

        probability = model.predict_proba(
            input_data
        )[0][1]


    # ======================================
    # Prediction Result
    # ======================================

    st.markdown("---")

    st.header("📊 Prediction Result")


    if prediction == 1:

        st.error(
            "🔴 Customer is predicted to DEFAULT"
        )

        risk_level = "HIGH"

    else:

        st.success(
            "🟢 Customer is predicted NOT TO DEFAULT"
        )

        risk_level = "LOW"


    # ======================================
    # Result Cards
    # ======================================

    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "Risk Level",
            risk_level
        )


    with col2:

        if probability is not None:

            st.metric(
                "Default Probability",
                f"{probability * 100:.2f}%"
            )

        else:

            st.metric(
                "Default Probability",
                "Not Available"
            )


    # ======================================
    # Probability Bar
    # ======================================

    if probability is not None:

        st.subheader("Default Probability")

        st.progress(
            float(probability)
        )


    # ======================================
    # Simple Risk Indicators
    # ======================================

    st.subheader("⚠ Risk Indicators")

    risk_indicators = []


    # Recent repayment delay

    if pay_0 > 0:

        risk_indicators.append(
            "Recent repayment delay detected."
        )


    # Maximum delay

    if max_delay >= 2:

        risk_indicators.append(
            "Significant repayment delay appears in the history."
        )


    # Payment behavior

    if pay_to_bill_ratio < 0.20:

        risk_indicators.append(
            "Average payment is low compared with the average bill."
        )


    # If no indicators

    if len(risk_indicators) == 0:

        st.success(
            "No major risk indicators detected from the entered data."
        )

    else:

        for indicator in risk_indicators:

            st.warning(indicator)


# ==========================================
# About Project
# ==========================================

st.markdown("---")


with st.expander("ℹ️ About This Project"):

    st.write(
        """
        Credit Card Default Risk Prediction is a machine
        learning project that predicts whether a customer
        is likely to default on their next credit card payment.

        The model uses:

        • Customer demographic information
        • Repayment history
        • Billing history
        • Payment history

        Models used during the project:

        • Logistic Regression
        • Decision Tree
        • Random Forest

        Model evaluation is performed using Stratified
        K-Fold Cross-Validation.
        """
    )
