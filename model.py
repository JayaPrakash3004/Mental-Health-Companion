import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv("stress_detection_data.csv")

# Select relevant features
features = [
    "Age", "Gender", "Occupation", "Marital_Status", "Sleep_Duration",
    "Sleep_Quality", "Physical_Activity", "Screen_Time", "Caffeine_Intake",
    "Alcohol_Intake", "Smoking_Habit", "Work_Hours", "Travel_Time",
    "Social_Interactions", "Meditation_Practice", "Exercise_Type",
    "Blood_Pressure", "Cholesterol_Level", "Blood_Sugar_Level"
]
target = "Stress_Detection"


label_encoders = {}
for col in ["Gender", "Occupation", "Marital_Status", "Smoking_Habit", "Meditation_Practice", "Exercise_Type"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  

target_encoder = LabelEncoder()
df[target] = target_encoder.fit_transform(df[target])


X = df[features]
y = df[target]

scaler = StandardScaler()
X = scaler.fit_transform(X)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)


joblib.dump(model, "stress_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoders, "encoders.pkl")
joblib.dump(target_encoder, "target_encoder.pkl")


def safe_encode(value, encoder, feature_name, default=-1):
    try:
        return encoder.transform([value])[0]
    except ValueError:
        print(f"⚠ Warning: '{value}' is an unseen category for '{feature_name}'. Assigning default value ({default}).")
        return default  


def predict_stress():
  
    model = joblib.load("stress_model.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoders = joblib.load("encoders.pkl")
    target_encoder = joblib.load("target_encoder.pkl")

    
    age = int(input("Enter Age: "))
    gender = input("Enter Gender (Male/Female): ")
    occupation = input("Enter Occupation: ")
    marital_status = input("Enter Marital Status (Single/Married/Divorced): ")
    sleep_duration = float(input("Enter Sleep Duration (hours): "))
    sleep_quality = float(input("Enter Sleep Quality (1-5): "))
    physical_activity = float(input("Enter Physical Activity (hours/day): "))
    screen_time = float(input("Enter Screen Time (hours/day): "))
    caffeine_intake = int(input("Enter Caffeine Intake (cups/day): "))
    alcohol_intake = int(input("Enter Alcohol Intake (drinks/week): "))
    smoking_habit = input("Do you smoke? (Yes/No): ")
    work_hours = int(input("Enter Work Hours per day: "))
    travel_time = float(input("Enter Travel Time (hours/day): "))
    social_interactions = int(input("Enter Social Interactions (hours/day): "))
    meditation_practice = input("Do you meditate? (Yes/No): ")
    exercise_type = input("Enter Exercise Type (Cardio/Yoga/Strength Training): ")
    blood_pressure = int(input("Enter Blood Pressure (e.g., 120): "))
    cholesterol_level = int(input("Enter Cholesterol Level: "))
    blood_sugar_level = int(input("Enter Blood Sugar Level: "))

    
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

    
    user_df = pd.DataFrame([user_data])
    user_scaled = scaler.transform(user_df)

    
    stress_prediction = model.predict(user_scaled)
    stress_level = target_encoder.inverse_transform(stress_prediction)[0]

    print(f"\n🧠 Predicted Stress Level: {stress_level}")


# Run prediction
predict_stress()