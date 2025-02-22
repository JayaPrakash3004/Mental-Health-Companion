import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import streamlit as st

# Cache the model, scaler, and encoders to prevent reloading
@st.cache_resource
def load_model():
    return {
        "model": joblib.load("stress_model.pkl"),
        "scaler": joblib.load("scaler.pkl"),
        "label_encoders": joblib.load("encoders.pkl"),
        "target_encoder": joblib.load("target_encoder.pkl")
    }

# Load the model once
model_data = load_model()

# Function to safely encode categorical inputs
def safe_encode(value, encoder, feature_name, default=-1):
    try:
        return encoder.transform([value])[0]
    except ValueError:
        st.warning(f"⚠ Warning: '{value}' is an unseen category for '{feature_name}'. Assigning default value ({default}).")
        return default

# Streamlit UI
st.title("🧠 Stress Level Prediction")
st.write("This app predicts your stress level based on various lifestyle factors.")

# User inputs (stored in session state to prevent reset)
age = st.number_input("Enter Age:", min_value=0, key="age")
gender = st.selectbox("Select Gender", ["Male", "Female"], key="gender")
occupation = st.text_input("Enter Occupation:", key="occupation")
marital_status = st.selectbox("Select Marital Status", ["Single", "Married", "Divorced"], key="marital_status")
sleep_duration = st.number_input("Enter Sleep Duration (hours):", min_value=0.0, key="sleep_duration")
sleep_quality = st.slider("Enter Sleep Quality (1-5):", min_value=1, max_value=5, key="sleep_quality")
physical_activity = st.number_input("Enter Physical Activity (hours/day):", min_value=0.0, key="physical_activity")
screen_time = st.number_input("Enter Screen Time (hours/day):", min_value=0.0, key="screen_time")
caffeine_intake = st.number_input("Enter Caffeine Intake (cups/day):", min_value=0, key="caffeine_intake")
alcohol_intake = st.number_input("Enter Alcohol Intake (drinks/week):", min_value=0, key="alcohol_intake")
smoking_habit = st.selectbox("Do you smoke?", ["Yes", "No"], key="smoking_habit")
work_hours = st.number_input("Enter Work Hours per day:", min_value=0, key="work_hours")
travel_time = st.number_input("Enter Travel Time (hours/day):", min_value=0.0, key="travel_time")
social_interactions = st.number_input("Enter Social Interactions (hours/day):", min_value=0, key="social_interactions")
meditation_practice = st.selectbox("Do you meditate?", ["Yes", "No"], key="meditation_practice")
exercise_type = st.selectbox("Enter Exercise Type", ["Cardio", "Yoga", "Strength Training"], key="exercise_type")
blood_pressure = st.number_input("Enter Blood Pressure (e.g., 120):", min_value=0, key="blood_pressure")
cholesterol_level = st.number_input("Enter Cholesterol Level:", min_value=0, key="cholesterol_level")
blood_sugar_level = st.number_input("Enter Blood Sugar Level:", min_value=0, key="blood_sugar_level")

# Prediction function
def predict_stress():
    model = model_data["model"]
    scaler = model_data["scaler"]
    label_encoders = model_data["label_encoders"]
    target_encoder = model_data["target_encoder"]

    # Prepare user data
    user_data = {
        "Age": age,
        "Gender": safe_encode(gender, label_encoders["Gender"], "Gender"),
        "Occupation": safe_encode(occupation, label_encoders["Occupation"], "Occupation"),
        "Marital_Status": safe_encode(marital_status, label_encoders["Marital_Status"], "Marital_Status"),
        "Sleep_Duration": sleep_duration,
        "Sleep_Quality": sleep_quality,
        "Physical_Activity": physical_activity,
        "Screen_Time": screen_time,
        "Caffeine_Intake": caffeine_intake,
        "Alcohol_Intake": alcohol_intake,
        "Smoking_Habit": safe_encode(smoking_habit, label_encoders["Smoking_Habit"], "Smoking_Habit"),
        "Work_Hours": work_hours,
        "Travel_Time": travel_time,
        "Social_Interactions": social_interactions,
        "Meditation_Practice": safe_encode(meditation_practice, label_encoders["Meditation_Practice"], "Meditation_Practice"),
        "Exercise_Type": safe_encode(exercise_type, label_encoders["Exercise_Type"], "Exercise_Type"),
        "Blood_Pressure": blood_pressure,
        "Cholesterol_Level": cholesterol_level,
        "Blood_Sugar_Level": blood_sugar_level
    }

    # Convert to DataFrame and scale
    user_df = pd.DataFrame([user_data])
    user_scaled = scaler.transform(user_df)

    # Predict stress level
    stress_prediction = model.predict(user_scaled)
    stress_level = target_encoder.inverse_transform(stress_prediction)[0]

    st.success(f"🧠 Predicted Stress Level: **{stress_level}**")

# Button to predict
if st.button("Predict Stress Level"):
    predict_stress()
