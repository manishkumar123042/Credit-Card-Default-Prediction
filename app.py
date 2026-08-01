import streamlit as st
import pandas as pd
import pickle

# ======================================================
# Load Model
# ======================================================

with open("Models/model.pkl", "rb") as file:
    model = pickle.load(file)

# ======================================================
# Page Configuration
# ======================================================

st.set_page_config(
    page_title="Credit Card Default Risk Prediction",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================
# Custom CSS
# ======================================================

st.markdown("""
<style>

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}

.card{
    padding:20px;
    border-radius:15px;
    border:1px solid rgba(128,128,128,0.25);
    margin-bottom:20px;
}

.title{
    font-size:36px;
    font-weight:700;
}

.subtitle{
    font-size:17px;
    opacity:0.8;
}

.section{
    font-size:24px;
    font-weight:600;
    margin-bottom:15px;
}

div[data-testid="stMetric"]{
    border:1px solid rgba(128,128,128,.25);
    border-radius:12px;
    padding:15px;
}

</style>
""", unsafe_allow_html=True)

# ======================================================
# Sidebar
# ======================================================

with st.sidebar:

    st.title("💳 Dashboard")

    st.markdown("---")

    st.success("Model : Random Forest")

    st.info("Evaluation : Stratified 5-Fold Cross Validation")

    st.write("### Features Used")

    st.write("• Customer Details")

    st.write("• Repayment History")

    st.write("• Bill Amount History")

    st.write("• Payment Amount History")

    st.markdown("---")

    st.caption("Credit Card Default Prediction")

# ======================================================
# Header
# ======================================================

st.markdown(
"""
<div class="title">
💳 Credit Card Default Risk Prediction
</div>

<div class="subtitle">
AI-Based Financial Risk Assessment System
</div>
""",
unsafe_allow_html=True
)

st.write("Predict whether a customer is likely to default on the next month's credit card payment.")

st.markdown("---")

# ======================================================
# CUSTOMER DETAILS CARD
# ======================================================

st.markdown(
"""
<div class="card">

<div class="section">

👤 Customer Details

</div>

</div>
""",
unsafe_allow_html=True
)

col1,col2 = st.columns(2)

with col1:

    limit_bal = st.number_input(
        "Credit Limit",
        min_value=10000,
        max_value=1000000,
        value=200000,
        step=10000
    )

    age = st.slider(
        "Age",
        21,
        80,
        30
    )

with col2:

    sex = st.selectbox(
        "Gender",
        [1,2],
        format_func=lambda x:
        "Male" if x==1 else "Female"
    )

    education = st.selectbox(
        "Education",
        [1,2,3,4],
        format_func=lambda x:{
            1:"Graduate School",
            2:"University",
            3:"High School",
            4:"Other"
        }[x]
    )

    marriage = st.selectbox(
        "Marital Status",
        [1,2,3],
        format_func=lambda x:{
            1:"Married",
            2:"Single",
            3:"Other"
        }[x]
    )

st.markdown("---")

# ======================================================
# REPAYMENT HISTORY CARD
# ======================================================

st.markdown(
"""
<div class="card">

<div class="section">

📊 Repayment History

</div>

</div>
""",
unsafe_allow_html=True
)

st.info(
"""
Repayment Status Codes

-2 = No Consumption

-1 = Paid Duly

0 = Use Revolving Credit

1–8 = Payment Delay (Months)
"""
)

r1,r2,r3 = st.columns(3)

with r1:

    pay_1 = st.selectbox(
        "Most Recent Month",
        list(range(-2,9)),
        index=2
    )

    pay_2 = st.selectbox(
        "2 Months Ago",
        list(range(-2,9)),
        index=2
    )

with r2:

    pay_3 = st.selectbox(
        "3 Months Ago",
        list(range(-2,9)),
        index=2
    )

    pay_4 = st.selectbox(
        "4 Months Ago",
        list(range(-2,9)),
        index=2
    )

with r3:

    pay_5 = st.selectbox(
        "5 Months Ago",
        list(range(-2,9)),
        index=2
    )

    pay_6 = st.selectbox(
        "6 Months Ago",
        list(range(-2,9)),
        index=2
    )

st.markdown("---")
# ======================================================
# BILL AMOUNT CARD
# ======================================================

st.markdown(
"""
<div class="card">

<div class="section">

💰 Bill Amount History

</div>

</div>
""",
unsafe_allow_html=True
)

st.caption("Outstanding bill amount for the last six months.")

b1, b2, b3 = st.columns(3)

with b1:

    bill_amt1 = st.number_input(
        "Current Month",
        min_value=0,
        value=5000,
        step=1000
    )

    bill_amt2 = st.number_input(
        "2 Months Ago",
        min_value=0,
        value=5000,
        step=1000
    )

with b2:

    bill_amt3 = st.number_input(
        "3 Months Ago",
        min_value=0,
        value=5000,
        step=1000
    )

    bill_amt4 = st.number_input(
        "4 Months Ago",
        min_value=0,
        value=5000,
        step=1000
    )

with b3:

    bill_amt5 = st.number_input(
        "5 Months Ago",
        min_value=0,
        value=5000,
        step=1000
    )

    bill_amt6 = st.number_input(
        "6 Months Ago",
        min_value=0,
        value=5000,
        step=1000
    )

st.markdown("---")

# ======================================================
# PAYMENT AMOUNT CARD
# ======================================================

st.markdown(
"""
<div class="card">

<div class="section">

💵 Payment Amount History

</div>

</div>
""",
unsafe_allow_html=True
)

st.caption("Amount paid by the customer during the last six months.")

p1, p2, p3 = st.columns(3)

with p1:

    pay_amt1 = st.number_input(
        "Payment 1",
        min_value=0,
        value=2000,
        step=500
    )

    pay_amt2 = st.number_input(
        "Payment 2",
        min_value=0,
        value=2000,
        step=500
    )

with p2:

    pay_amt3 = st.number_input(
        "Payment 3",
        min_value=0,
        value=2000,
        step=500
    )

    pay_amt4 = st.number_input(
        "Payment 4",
        min_value=0,
        value=2000,
        step=500
    )

with p3:

    pay_amt5 = st.number_input(
        "Payment 5",
        min_value=0,
        value=2000,
        step=500
    )

    pay_amt6 = st.number_input(
        "Payment 6",
        min_value=0,
        value=2000,
        step=500
    )

st.markdown("---")

# ======================================================
# FEATURE ENGINEERING
# ======================================================

avg_bill_amt = (
    bill_amt1 +
    bill_amt2 +
    bill_amt3 +
    bill_amt4 +
    bill_amt5 +
    bill_amt6
) / 6

avg_pay_amt = (
    pay_amt1 +
    pay_amt2 +
    pay_amt3 +
    pay_amt4 +
    pay_amt5 +
    pay_amt6
) / 6

pay_to_bill_ratio = avg_pay_amt / (avg_bill_amt + 1)

max_delay = max(
    pay_1,
    pay_2,
    pay_3,
    pay_4,
    pay_5,
    pay_6
)

# ======================================================
# ENGINEERED FEATURES
# ======================================================

st.subheader("📈 Financial Summary")

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric(
        "Average Bill",
        f"₹{avg_bill_amt:,.0f}"
    )

with m2:
    st.metric(
        "Average Payment",
        f"₹{avg_pay_amt:,.0f}"
    )

with m3:
    st.metric(
        "Payment Ratio",
        f"{pay_to_bill_ratio:.2f}"
    )

with m4:
    st.metric(
        "Maximum Delay",
        max_delay
    )

st.markdown("---")

# ======================================================
# PREDICT BUTTON
# ======================================================

predict = st.button(
    "🔍 Predict Default Risk",
    use_container_width=True,
    type="primary"
)

if predict:

    input_df = pd.DataFrame({

        "ID":[1],

        "LIMIT_BAL":[limit_bal],

        "SEX":[sex],

        "EDUCATION":[education],

        "MARRIAGE":[marriage],

        "AGE":[age],

        "PAY_1":[pay_1],
        "PAY_2":[pay_2],
        "PAY_3":[pay_3],
        "PAY_4":[pay_4],
        "PAY_5":[pay_5],
        "PAY_6":[pay_6],

        "BILL_AMT1":[bill_amt1],
        "BILL_AMT2":[bill_amt2],
        "BILL_AMT3":[bill_amt3],
        "BILL_AMT4":[bill_amt4],
        "BILL_AMT5":[bill_amt5],
        "BILL_AMT6":[bill_amt6],

        "PAY_AMT1":[pay_amt1],
        "PAY_AMT2":[pay_amt2],
        "PAY_AMT3":[pay_amt3],
        "PAY_AMT4":[pay_amt4],
        "PAY_AMT5":[pay_amt5],
        "PAY_AMT6":[pay_amt6],

        "avg_bill_amt":[avg_bill_amt],
        "avg_pay_amt":[avg_pay_amt],
        "pay_to_bill_ratio":[pay_to_bill_ratio],
        "max_delay":[max_delay]

    })

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0][1]

    # ======================================================
# PREDICTION RESULT
# ======================================================

    st.markdown("---")

    st.header("📊 Prediction Dashboard")

    probability_percent = probability * 100

    st.subheader("Default Probability")

    st.progress(float(probability))

    st.write(f"**Probability : {probability_percent:.2f}%**")

    st.markdown("---")

# ======================================================
# RISK STATUS
# ======================================================

    if prediction == 0:

        st.success("✅ Prediction : Customer is NOT likely to default.")

    else:

        st.error("⚠️ Prediction : Customer is likely to default.")

# ======================================================
# RISK LEVEL
# ======================================================

    st.subheader("Risk Level")

    if probability < 0.30:

        st.success("🟢 LOW RISK")

    elif probability < 0.60:

        st.warning("🟡 MEDIUM RISK")

    else:

        st.error("🔴 HIGH RISK")

# ======================================================
# BUSINESS RECOMMENDATION
# ======================================================

    st.markdown("---")

    st.subheader("💡 Recommendation")

    if probability < 0.30:

        st.success("""
### Recommended Action

✔ Continue normal credit limit

✔ Customer shows healthy repayment behaviour

✔ No immediate action required
""")

    elif probability < 0.60:

        st.warning("""
### Recommended Action

✔ Monitor repayment behaviour

✔ Send payment reminder

✔ Review customer account regularly
""")

    else:

        st.error("""
### Recommended Action

✔ High probability of default

✔ Review customer's credit profile

✔ Consider reducing credit limit

✔ Offer EMI or repayment plan

✔ Contact customer before next billing cycle
""")

# ======================================================
# CUSTOMER SUMMARY
# ======================================================

    st.markdown("---")

    st.subheader("📋 Customer Summary")

    summary = pd.DataFrame({

        "Feature":[

            "Credit Limit",
            "Age",
            "Gender",
            "Education",
            "Marriage",

            "Average Bill Amount",

            "Average Payment Amount",

            "Payment/Bill Ratio",

            "Maximum Delay"

        ],

        "Value":[

            f"₹{limit_bal:,}",

            age,

            "Male" if sex==1 else "Female",

            {
                1:"Graduate School",
                2:"University",
                3:"High School",
                4:"Other"
            }[education],

            {
                1:"Married",
                2:"Single",
                3:"Other"
            }[marriage],

            f"₹{avg_bill_amt:,.2f}",

            f"₹{avg_pay_amt:,.2f}",

            round(pay_to_bill_ratio,2),

            max_delay

        ]

    })

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )

# ======================================================
# MODEL INFORMATION
# ======================================================

    st.markdown("---")

    st.subheader("🤖 Model Information")

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Model",
            "Random Forest"
        )

    with c2:

        st.metric(
            "Evaluation",
            "5-Fold CV"
        )

    with c3:

        st.metric(
            "Prediction",
            "Completed"
        )
if st.button("Reset Inputs"):
    st.rerun()
# ======================================================
# THANK YOU
# ======================================================

with st.expander("Model Information"):

    st.write("""
Algorithm : Random Forest

Validation : Stratified 5-Fold Cross Validation

Dataset : UCI Credit Card Default

Target : Default Payment Next Month
""")