import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# Title
# -----------------------------
st.set_page_config(page_title="Loan Prediction App", layout="centered")
st.title("💰 Intelligent Loan Approval Prediction")
st.write("Fill the details below to check whether the loan will be approved or rejected.")

# -----------------------------
# Sample Dataset (Training)
# -----------------------------
data = {
    'Age': [25, 35, 45, 32, 23, 40, 60, 48, 33, 28],
    'Income': [30000, 60000, 80000, 50000, 25000, 90000, 100000, 75000, 62000, 40000],
    'CreditScore': [650, 700, 750, 680, 620, 800, 820, 770, 710, 690],
    'LoanAmount': [20000, 25000, 30000, 22000, 15000, 40000, 50000, 35000, 27000, 21000],
    'LoanTerm': [2, 3, 5, 3, 2, 6, 7, 5, 4, 3],
    'Approved': [0, 1, 1, 1, 0, 1, 1, 1, 1, 0]
}

df = pd.DataFrame(data)

X = df.drop('Approved', axis=1)
y = df['Approved']

# -----------------------------
# Train Model
# -----------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# -----------------------------
# User Input Section
# -----------------------------
st.subheader("📋 Enter Customer Details")

age = st.slider("Age", 18, 70, 30)
income = st.number_input("Income (₹)", min_value=10000, max_value=200000, value=50000)
credit = st.slider("Credit Score", 300, 900, 700)
loan_amount = st.number_input("Loan Amount (₹)", min_value=5000, max_value=100000, value=20000)
loan_term = st.slider("Loan Term (years)", 1, 10, 3)

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("🔍 Predict Loan Status"):
    input_data = np.array([[age, income, credit, loan_amount, loan_term]])
    
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    st.subheader("📊 Result")

    if prediction[0] == 1:
        st.success(f"✅ Loan Approved")
    else:
        st.error(f"❌ Loan Rejected")

    st.write(f"**Confidence Score:** {max(probability[0]):.2f}")

# -----------------------------
# Feature Importance
# -----------------------------
st.subheader("📈 Feature Importance")

importance = model.feature_importances_
features = X.columns

importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importance
}).sort_values(by="Importance", ascending=False)

st.bar_chart(importance_df.set_index('Feature'))