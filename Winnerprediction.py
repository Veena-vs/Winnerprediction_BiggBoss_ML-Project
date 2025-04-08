import streamlit as st
import pandas as pd
import joblib  # Use pickle if that’s what you used to save the model

# Load your trained model
model = joblib.load("winner_prediction_model.pkl")  # Update with your model filename

# Set up page
st.set_page_config(page_title="Bigg Boss Winner Predictor", page_icon="👑", layout="centered")
st.title("👑 Bigg Boss Winner Predictor")
st.markdown("Predict whether a contestant has a chance of winning Bigg Boss based on their stats!")

# User inputs
season = st.selectbox("Season", list(range(1, 17)))
gender = st.radio("Gender", ["Male", "Female", "Other"])
age = st.slider("Age", min_value=18, max_value=65, value=30)
total_votes = st.number_input("Total Votes Received", min_value=0, value=10000)
nominations = st.number_input("Number of Nominations", min_value=0, value=5)
captaincies = st.number_input("Times as Captain", min_value=0, value=1)
tasks_won = st.number_input("Tasks Won", min_value=0, value=3)
fights = st.number_input("Fights Involved In", min_value=0, value=2)

# Encode categorical values manually (if not using encoders)
gender_encoded = {"Male": 0, "Female": 1, "Other": 2}[gender]

# Create DataFrame for prediction
input_data = pd.DataFrame({
    "season": [season],
    "gender": [gender_encoded],
    "age": [age],
    "total_votes": [total_votes],
    "nominations": [nominations],
    "captaincies": [captaincies],
    "tasks_won": [tasks_won],
    "fights": [fights]
})

# Prediction
if st.button("Predict Winner"):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]  # Probability of winning

    if prediction[0] == 1:
        st.success("🏆 This contestant is likely to be the **WINNER**!")
    else:
        st.info("🤔 This contestant might **not win** the show.")

    st.markdown(f"**Winning Probability:** `{probability * 100:.2f}%`")

# Footer
st.markdown("---")
st.caption("Built by Veena | Powered by Streamlit & Machine Learning 🧠")
