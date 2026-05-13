import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime

# -----------------------------------
# CONFIG
# -----------------------------------
st.set_page_config(
    page_title="AI No-Show Prediction System",
    page_icon="🏥",
    layout="wide"
)

# -----------------------------------
# LOAD MODEL
# -----------------------------------
model = joblib.load("noshow_model.pkl")
feature_names = joblib.load("feature_names.pkl")

# -----------------------------------
# LOAD DATA
# -----------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("MedicalAppointment.csv")
    df['NoShow'] = df['No-show'].map({'Yes':1,'No':0})
    return df

df = load_data()

# -----------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("", ["📊 Dashboard", "🔮 Prediction", "📂 Batch Prediction"])

# ==========================================
# 📊 DASHBOARD PAGE
# ==========================================
if page == "📊 Dashboard":

    st.title("📊 Healthcare No-Show Dashboard")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Appointments", len(df))
    col2.metric("No-Show Rate", f"{df['NoShow'].mean()*100:.2f}%")
    col3.metric("Avg Age", f"{df['Age'].mean():.1f}")

    st.divider()

    # Filters
    st.subheader("🔍 Filters")
    age_range = st.slider("Age Range", 0, 100, (10, 60))
    filtered = df[(df['Age'] >= age_range[0]) & (df['Age'] <= age_range[1])]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("No-Show by Gender")
        st.bar_chart(filtered.groupby("Gender")["NoShow"].mean())

    with col2:
        st.subheader("No-Show by SMS")
        st.bar_chart(filtered.groupby("SMS_received")["NoShow"].mean())

    st.divider()

    # Highlight high-risk (actual no-show)
    st.subheader("⚠️ High-Risk Patients (Historical)")
    high_risk = filtered[filtered["NoShow"] == 1]

    st.dataframe(high_risk.sample(min(20, len(high_risk))), use_container_width=True)

# ==========================================
# 🔮 SINGLE PREDICTION PAGE
# ==========================================
elif page == "🔮 Prediction":

    st.title("🔮 Patient Risk Prediction")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 0, 100, 30)
        lead_time = st.number_input("Lead Time (days)", 0, 60, 10)
        prior_noshows = st.number_input("Prior No-Shows", 0, 10, 0)
        gender = st.selectbox("Gender", ["Female", "Male"])

    with col2:
        sms = st.selectbox("SMS Received", [0, 1])
        chronic = st.selectbox("Chronic Condition", [0, 1])
        email = st.text_input("Email")
        phone = st.text_input("Phone")

    # ----------- PREPROCESSING -----------
    gender = 1 if gender == "Male" else 0
    today = datetime.datetime.today()

    input_data = np.array([[
        gender,
        age,
        0, 0, 0, 0, 0,
        sms,
        lead_time,
        prior_noshows,
        today.weekday(),
        today.month,
        chronic,
        0.2  # avg neighbourhood risk
    ]])

    if st.button("Predict Risk"):

        prob = model.predict_proba(input_data)[0][1]

        # Threshold tuning
        if prob > 0.65:
            risk = "HIGH"
            color = "red"
        elif prob > 0.4:
            risk = "MEDIUM"
            color = "orange"
        else:
            risk = "LOW"
            color = "green"

        st.subheader("Prediction Result")
        st.metric("No-Show Probability", f"{prob:.2f}")
        st.markdown(f"### Risk Level: :{color}[{risk}]")

        # ----------- ACTION ENGINE -----------
        st.subheader("Recommended Action")

        if risk == "HIGH":
            st.error("Call patient + Send SMS immediately")
        elif risk == "MEDIUM":
            st.warning("Send reminder notification")
        else:
            st.success("No action needed")

# ==========================================
# 📂 BATCH PREDICTION (REAL WORLD)
# ==========================================
else:

    st.title("📂 Batch Prediction System")

    uploaded_file = st.file_uploader("Upload CSV file")

    if uploaded_file:

        data = pd.read_csv(uploaded_file)

        st.write("Uploaded Data Preview")
        st.dataframe(data.head())

        # ----------- BASIC PREPROCESSING -----------
        data["Gender"] = (data["Gender"] == "M").astype(int)

        data["lead_time"] = 10
        data["prior_noshows"] = 0
        data["day_of_week"] = 2
        data["appointment_month"] = 5
        data["has_chronic"] = 0
        data["neighbourhood_risk"] = 0.2

        features = data[feature_names]

        probs = model.predict_proba(features)[:,1]
        data["Risk_Prob"] = probs

        data["Risk_Level"] = pd.cut(
            probs,
            bins=[0,0.4,0.65,1],
            labels=["Low","Medium","High"]
        )

        st.subheader("Prediction Results")
        st.dataframe(data, use_container_width=True)

        # Highlight high risk
        st.subheader("⚠️ High Risk Patients")
        st.dataframe(data[data["Risk_Level"] == "High"])
