
import streamlit as st
import joblib

# Load the trained model and label encoder
model = joblib.load('rugby_prediction_model.joblib')
label_encoder = joblib.load('rugby_label_encoder.joblib')

# Title of the app
st.title("Rugby Match Result Difference Predictor")

# Input fields
team_1 = st.selectbox('Select Team 1:', label_encoder.classes_)
team_2 = st.selectbox('Select Team 2:', label_encoder.classes_)
team_1_win_odds = st.number_input('Enter Team 1 Win Odds:', min_value=0.0, value=1.0)
draw_odds = st.number_input('Enter Draw Odds:', min_value=0.0, value=1.0)
team_2_win_odds = st.number_input('Enter Team 2 Win Odds:', min_value=0.0, value=1.0)

# Predict button
if st.button('Predict'):
    # Encoding the selected teams
    team_1_encoded = label_encoder.transform([team_1])[0]
    team_2_encoded = label_encoder.transform([team_2])[0]
    
    # Creating a feature vector
    feature_vector = [[team_1_encoded, team_2_encoded, team_1_win_odds, draw_odds, team_2_win_odds]]
    
    # Predicting the result difference using the trained model
    result_difference = model.predict(feature_vector)[0]
    
    # Displaying the predicted result difference
    st.write(f'The predicted result difference is: {result_difference:.2f}')

# Instructions on how to run the app
st.write('')
st.write('**Instructions:**')
st.write('1. Save this script as a `.py` file.')
st.write('2. Open a terminal and navigate to the directory where the script is located.')
st.write('3. Run the command `streamlit run script_name.py` (replace "script_name.py" with the actual script name).')
st.write('4. The app will open in your web browser.')
