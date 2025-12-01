import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Workout Type Predictor", page_icon="💪", layout="centered")

st.title("🏋‍♀ Workout Type Prediction App")
st.write("This app predicts your ideal *Workout Type* based on your fitness attributes.")

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\RUBAINA\Downloads\Final_data (1).csv")
    return df

data = load_data()

# --- Prepare Data + Model ---
@st.cache_resource
def train_model(data):
    drop_cols = [
        'Name of Exercise', 'Burns_Calories_Bin', 'Target Muscle Group', 'Equipment Needed',
        'Benefit', 'Body Part', 'Type of Muscle', 'Burns Calories (per 30 min)',
        'Burns Calories (per 30 min)_bc', 'Fat_Percentage_bc', 'meal_name', 'cooking_method',
        'meal_type', 'pct_carbs', 'serving_size_g', 'cholesterol_mg', 'prep_time_min',
        'cook_time_min', 'rating', 'Sets', 'Reps'
    ]
    data = data.drop(columns=[c for c in drop_cols if c in data.columns], errors='ignore')

    le = LabelEncoder()
    data['Workout_Type'] = le.fit_transform(data['Workout_Type'])

    X = data.drop('Workout_Type', axis=1)
    y = data['Workout_Type']

    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        X[col] = LabelEncoder().fit_transform(X[col])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_scaled, y)

    return model, scaler, le, categorical_cols, X, data

model, scaler, le, categorical_cols, X, data = train_model(data)

# --- User Input ---
st.header("Enter Your Details")

user_input = {}
for col in X.columns:
    if col in categorical_cols:
        options = sorted(list(data[col].unique()))
        user_input[col] = st.selectbox(f"{col}", options)
    else:
        min_val, max_val = float(data[col].min()), float(data[col].max())
        mean_val = float(data[col].mean())
        user_input[col] = st.slider(f"{col}", min_val, max_val, mean_val)

user_df = pd.DataFrame([user_input])

# Encode categorical
for col in categorical_cols:
    encoder = LabelEncoder()
    encoder.fit(data[col])
    user_df[col] = encoder.transform(user_df[col])

# Scale
user_scaled = scaler.transform(user_df)

# --- Prediction ---
if st.button("Predict Workout Type 💥"):
    pred = model.predict(user_scaled)
    pred_label = le.inverse_transform(pred)[0]
    st.success(f"🏆 Recommended Workout Type: *{pred_label}*")


st.markdown("---")
st.caption("Developed with ❤ using Streamlit and scikit-learn")