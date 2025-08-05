import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained model and dataset
model = pickle.load(open("model.pkl", "rb"))
data = pd.read_csv("dataset.csv")

st.title("🤖 MedLens - AI Symptom Checker")

# Get symptom list from dataset
symptoms = list(data.columns[:-1])
selected_symptoms = st.multiselect("Select symptoms", symptoms)

if st.button("Predict Disease"):
    input_data = [1 if symptom in selected_symptoms else 0 for symptom in symptoms]
    prediction = model.predict([input_data])[0]
    st.success(f"Possible Disease: **{prediction}**")
