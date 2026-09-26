# app.py
# Streamlit front-end for the "Visit with Us" Wellness Tourism Package
# purchase-prediction model. Loads the trained pipeline (bundled locally,
# falling back to the Hugging Face Model Hub) and serves live predictions.

import os
import joblib
import pandas as pd
import streamlit as st

HF_USERNAME = "jashangrover324"         
MODEL_REPO_ID = f"{HF_USERNAME}/tourism-wellness-model"
LOCAL_MODEL_PATH = "best_model.joblib"


@st.cache_resource
def load_model():
    # Prefer a locally bundled model (present alongside the app); otherwise
    # download the latest registered version from the Hugging Face Hub.
    if os.path.exists(LOCAL_MODEL_PATH):
        return joblib.load(LOCAL_MODEL_PATH)
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=MODEL_REPO_ID, filename="best_model.joblib")
    return joblib.load(path)


st.set_page_config(page_title="Wellness Package Predictor", page_icon="🧳")
st.title("🧳 Visit with Us -- Wellness Package Purchase Predictor")
st.write(
    "Enter a customer's profile and sales-interaction details to predict "
    "the likelihood they will purchase the new Wellness Tourism Package."
)

model = load_model()

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 18, 100, 35)
    type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    occupation = st.selectbox(
        "Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"]
    )
    gender = st.selectbox("Gender", ["Male", "Female"])
    num_persons = st.number_input("Number of Persons Visiting", 1, 10, 2)
    num_followups = st.number_input("Number of Followups", 0, 10, 3)
    product_pitched = st.selectbox(
        "Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
    )
    preferred_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])

with col2:
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    num_trips = st.number_input("Number of Trips per Year", 0, 25, 3)
    passport = st.selectbox("Holds Passport?", ["Yes", "No"])
    pitch_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    own_car = st.selectbox("Owns a Car?", ["Yes", "No"])
    num_children = st.number_input("Number of Children Visiting", 0, 5, 0)
    designation = st.selectbox(
        "Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
    )
    monthly_income = st.number_input("Monthly Income", 1000, 100000, 22000)
    duration_of_pitch = st.number_input("Duration of Pitch (minutes)", 1, 60, 15)

if st.button("Predict"):
    input_df = pd.DataFrame([{
        "Age": age,
        "TypeofContact": type_of_contact,
        "CityTier": city_tier,
        "DurationOfPitch": duration_of_pitch,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": num_persons,
        "NumberOfFollowups": num_followups,
        "ProductPitched": product_pitched,
        "PreferredPropertyStar": preferred_star,
        "MaritalStatus": marital_status,
        "NumberOfTrips": num_trips,
        "Passport": 1 if passport == "Yes" else 0,
        "PitchSatisfactionScore": pitch_score,
        "OwnCar": 1 if own_car == "Yes" else 0,
        "NumberOfChildrenVisiting": num_children,
        "Designation": designation,
        "MonthlyIncome": monthly_income,
    }])

    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0, 1]

    if pred == 1:
        st.success(f"Likely to purchase the Wellness Package (probability: {proba:.1%})")
    else:
        st.warning(f"Unlikely to purchase the Wellness Package (probability: {proba:.1%})")
