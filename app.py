from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "churn_pipeline.joblib"

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")
st.title("📉 Customer Churn Predictor")
st.caption("Portfolio demo — predictions are illustrative and depend on the training data.")

if not MODEL_PATH.exists():
    st.warning("Model not found. Run `python train.py` first.")
    st.stop()

model = joblib.load(MODEL_PATH)
with st.form("customer_form"):
    st.subheader("Customer profile")
    tenure = st.slider("Tenure (months)", 0, 72, 18)
    monthly_charges = st.slider("Monthly charges", 10.0, 150.0, 70.0, 1.0)
    total_charges = st.number_input("Total charges", min_value=0.0, value=float(tenure * monthly_charges))
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
    support = st.selectbox("Tech support", ["Yes", "No", "No internet service"])
    payment = st.selectbox("Payment method", [
        "Electronic check", "Mailed check", "Bank transfer", "Credit card"
    ])
    paperless = st.selectbox("Paperless billing", ["Yes", "No"])
    submitted = st.form_submit_button("Predict churn risk")

if submitted:
    row = pd.DataFrame([{
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Contract": contract,
        "InternetService": internet,
        "TechSupport": support,
        "PaymentMethod": payment,
        "PaperlessBilling": paperless,
    }])
    # Keep feature names aligned with the included demo data.
    probability = float(model.predict_proba(row)[0, 1])
    st.metric("Estimated churn probability", f"{probability:.1%}")
    if probability >= 0.5:
        st.error("Higher predicted churn risk")
    else:
        st.success("Lower predicted churn risk")
    st.caption("This demo is not a validated business decision tool.")
