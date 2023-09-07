import streamlit as st
import joblib

# Load the trained model
model = joblib.load('final_model.pkl')

# Mapping of country names to numbers
country_mapping = {
    "England": 1,
    "New Zealand": 2,
    "Wales": 3,
    "Japan": 4,
    "USA": 5,
    "Ireland": 6,
    "Australia": 7,
    "Scotland": 8,
    "Argentina": 9,
    "South Africa": 10,
    "France": 11,
    "Georgia": 12,
    "Italy": 13,
    "Fiji": 14,
    "Russia": 15,
    "Samoa": 16,
    "Tonga": 17,
    "Namibia": 18,
    "Canada": 19,
    "Uruguay": 20,
    "Romania": 21
}

# Create input elements for the user to provide T1, T2, T1Odds, DrawOdds, and T2Odds
st.title("Predictive Model App")
T1 = st.selectbox("Select Team 1:", list(country_mapping.keys()))
T2 = st.selectbox("Select Team 2:", list(country_mapping.keys()))
T1Odds = st.number_input("Enter T1 Odds:")
DrawOdds = st.number_input("Enter Draw Odds:")
T2Odds = st.number_input("Enter T2 Odds:")

# Button to trigger prediction
if st.button("Predict"):
    # Prepare the feature vector and make a prediction
    features = [[country_mapping[T1], country_mapping[T2], T1Odds, DrawOdds, T2Odds]]
    prediction = model.predict(features)
    
    # Calculate the difference (not sure how you want to calculate this, please adjust)
    difference = prediction[0] - T1Odds  # Adjust this calculation as needed
    
    # Display the prediction and difference
    st.write(f"Prediction: {prediction[0]}")
    st.write(f"Difference: {difference}")
