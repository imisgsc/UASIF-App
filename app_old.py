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
st.title("⚽ UASIF — Athlete Injury Risk Dashboard")
st.markdown("**Unified AI-Driven Sports Intelligence Framework** · IC-ASSPT 2026")
st.divider()

# Sidebar inputs
st.sidebar.header("Enter Athlete Data")

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

# Main dashboard
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Injury Risk Score", f"{risk_percent}%")

with col2:
    st.metric("ACWR", round(acwr, 3))

with col3:
    if risk_percent < 40:
        st.success("🟢 LOW RISK — Train Normally")
    elif risk_percent < 65:
        st.warning("🟡 MODERATE RISK — Reduce Load")
    else:
        st.error("🔴 HIGH RISK — Rest Recommended")

st.divider()

# Athlete summary
st.subheader("Athlete Profile Summary")
col4, col5 = st.columns(2)

with col4:
    st.write(f"**Age:** {age} years")
    st.write(f"**Weight:** {weight} kg")
    st.write(f"**Height:** {height} cm")
    st.write(f"**Previous Injuries:** {previous_injuries}")

with col5:
    st.write(f"**Training Intensity:** {training_intensity}")
    st.write(f"**Recovery Time:** {recovery_time} days")
    st.write(f"**ACWR:** {round(acwr, 3)}")
    st.write(f"**Model Accuracy:** 85.5%")

st.divider()

# Feature importance chart
st.subheader("What Factors Drive Injury Risk?")
img = plt.imread('feature_importance.png')
st.image(img, width=600)

st.divider()
st.caption("UASIF Prototype · SRM Institute of Science and Technology · IC-ASSPT 2026")

