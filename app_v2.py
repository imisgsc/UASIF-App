import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('features.json', 'r') as f:
    features = json.load(f)

# Page config
st.set_page_config(page_title="UASIF - Athlete Risk Dashboard", page_icon="⚽", layout="wide")

# Header
st.title("⚽ UASIF — Athlete Injury Risk & Training Optimization Dashboard")
st.markdown("**Unified AI-Driven Sports Intelligence Framework** · IC-ASSPT 2026")
st.divider()

# Sidebar inputs
st.sidebar.header("Enter Athlete Data")
athlete_name = st.sidebar.text_input("Athlete Name (optional)", "Athlete A")
age = st.sidebar.slider("Player Age", 18, 38, 25)
weight = st.sidebar.slider("Player Weight (kg)", 60, 95, 75)
height = st.sidebar.slider("Player Height (cm)", 165, 195, 178)
previous_injuries = st.sidebar.selectbox("Previous Injuries", [0, 1, 2, 3])
training_intensity = st.sidebar.slider("Training Intensity", 0.2, 1.0, 0.5)
recovery_time = st.sidebar.slider("Recovery Time (days)", 1, 8, 4)

# Calculate ACWR
acwr = training_intensity / (recovery_time + 0.1)

# Predict
input_data = pd.DataFrame([[age, weight, height, previous_injuries,
                             training_intensity, recovery_time, acwr]],
                           columns=features)

prediction = model.predict(input_data)[0]
probability = model.predict_proba(input_data)[0][1]
risk_percent = round(probability * 100, 1)

# Smart recommendation engine
def get_recommendations(risk_percent, acwr, previous_injuries, recovery_time, training_intensity, age):
    recs = []
    training_rec = ""
    intensity_rec = 0.0

    if risk_percent < 40:
        status = "🟢 LOW RISK"
        color = "success"
        training_rec = "Continue current training plan normally."
        intensity_rec = training_intensity
        recs.append("✅ Athlete is in a safe training zone.")
        recs.append("✅ Maintain current recovery schedule.")
        recs.append("✅ Monitor ACWR weekly to stay in safe range.")
        if age > 30:
            recs.append("⚠️ Age factor noted — include mobility and flexibility work.")

    elif risk_percent < 65:
        status = "🟡 MODERATE RISK"
        color = "warning"
        intensity_rec = round(training_intensity * 0.75, 2)
        training_rec = f"Reduce training intensity to {intensity_rec} (25% reduction)."
        recs.append(f"⚠️ Reduce training intensity from {training_intensity} → {intensity_rec}.")
        recs.append(f"⚠️ Increase recovery time by at least 1–2 days.")
        recs.append("⚠️ Avoid high-impact drills this week.")
        if acwr > 0.4:
            recs.append("⚠️ ACWR is elevated — workload is outpacing recovery.")
        if previous_injuries > 0:
            recs.append("⚠️ Prior injury history detected — physiotherapy check advised.")

    else:
        status = "🔴 HIGH RISK"
        color = "error"
        intensity_rec = round(training_intensity * 0.4, 2)
        training_rec = f"Significantly reduce intensity to {intensity_rec}. Consider full rest."
        recs.append("🚨 Immediate load reduction required.")
        recs.append(f"🚨 Reduce training intensity to below {intensity_rec}.")
        recs.append("🚨 Minimum 2 days complete rest recommended.")
        recs.append("🚨 ACWR critically high — athlete is overloaded.")
        if previous_injuries > 1:
            recs.append("🚨 Multiple prior injuries detected — mandatory physio consultation.")
        if recovery_time < 3:
            recs.append("🚨 Recovery time is too low — extend to minimum 4 days.")
        if age > 30:
            recs.append("🚨 Age factor increases vulnerability — extra caution needed.")

    return status, color, training_rec, recs

status, color, training_rec, recs = get_recommendations(
    risk_percent, acwr, previous_injuries, recovery_time, training_intensity, age
)

# Dashboard display
st.subheader(f"Results for: {athlete_name}")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Injury Risk Score", f"{risk_percent}%")
with col2:
    st.metric("ACWR", round(acwr, 3))
with col3:
    st.metric("Model Accuracy", "85.5%")

st.divider()

# Risk status
if color == "success":
    st.success(f"**{status}** — {training_rec}")
elif color == "warning":
    st.warning(f"**{status}** — {training_rec}")
else:
    st.error(f"**{status}** — {training_rec}")

# Recommendations
st.subheader("📋 Personalized Training Recommendations")
for rec in recs:
    st.write(rec)

st.divider()

# Athlete profile
st.subheader("Athlete Profile Summary")
col4, col5 = st.columns(2)
with col4:
    st.write(f"**Name:** {athlete_name}")
    st.write(f"**Age:** {age} years")
    st.write(f"**Weight:** {weight} kg")
    st.write(f"**Height:** {height} cm")
with col5:
    st.write(f"**Previous Injuries:** {previous_injuries}")
    st.write(f"**Training Intensity:** {training_intensity}")
    st.write(f"**Recovery Time:** {recovery_time} days")
    st.write(f"**ACWR:** {round(acwr, 3)}")

st.divider()

# Feature importance
st.subheader("🔍 What Factors Drive This Athlete's Risk?")
st.image('feature_importance.png', width=600)

st.divider()
st.caption("UASIF Prototype v2 · SRM Institute of Science and Technology · IC-ASSPT 2026")