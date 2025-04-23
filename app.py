
import streamlit as st
import pandas as pd
import joblib
import os
import base64

def set_background(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/avif;base64,{encoded_string}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

st.set_page_config(page_title="Bigg Boss Winner Prediction", page_icon="🏆")
st.title("🏆 Bigg Boss Winner Prediction App")

# Apply background
set_background("C:/Users/DELL/Documents/oct_ml/Project/pexels-n-voitkevich-6532373.jpg")

# Load the trained model
model = joblib.load("winner_prediction_model2.pkl")


# Load label encoders
gender_encoder = joblib.load('encoders/gender_encoder.pkl')
wildcard_encoder = joblib.load('encoders/wild_card_encoder.pkl')
#language_encoder = joblib.load('encoders/language_encoder.pkl')
profession_encoder = joblib.load('encoders/profession_encoder.pkl')
state_encoder = joblib.load('encoders/most_viewed_states_encoder.pkl')
location_encoder = joblib.load('encoders/house_location_encoder.pkl')
ott_encoder = joblib.load('encoders/ott_season_encoder.pkl')

# Age Category Mapping
age_map = {'Young': 0, 'Middle': 1, 'Old': 2}
age_category_list = list(age_map.keys())

# Finalist Mapping
finalist_map = {'Yes': 1, 'No': 0}




# Streamlit UI
#st.set_page_config(page_title="Bigg Boss Winner Prediction", page_icon="🏆")
# st.title("🏆 Bigg Boss Winner Prediction App")

st.markdown("""
Welcome to the **Bigg Boss Winner Prediction App**!  
Fill in the contestant's details to predict their chances of winning the show.
""")


# Input Fields
profession = st.selectbox("Profession", profession_encoder.classes_)
gender = st.selectbox("Gender", gender_encoder.classes_)
#language = st.selectbox("Language", language_encoder.classes_)
age_category = st.selectbox("Age Category", age_category_list)
wild_card = st.selectbox("Wild Card Entry", wildcard_encoder.classes_)
most_viewed_state = st.selectbox("Most Viewed State", state_encoder.classes_)
house_location = st.selectbox("House Location", location_encoder.classes_)
ott_season = st.selectbox("OTT Season", ott_encoder.classes_)
num_evictions = st.slider("Number of Evictions Faced", 0, 10, 1)
finalist = st.selectbox("Is Finalist?", list(finalist_map.keys()))

# Create input data
input_data = pd.DataFrame([{
    'Profession': profession_encoder.transform([profession])[0],
    'Gender': gender_encoder.transform([gender])[0],
    #'Language': language_encoder.transform([language])[0],
    'Age': age_map[age_category],
    'Wild Card': wildcard_encoder.transform([wild_card])[0],
    'Most Viewed States': state_encoder.transform([most_viewed_state])[0],
    'House Location': location_encoder.transform([house_location])[0],
    'OTT Season': ott_encoder.transform([ott_season])[0],
    'Number of Evictions Faced': num_evictions,
    'Finalist': finalist_map[finalist]
}])

st.write("🔍 Input Data Sent to Model:")
st.write(input_data)


# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    st.success(f"🔏 Prediction: {'Winner' if prediction == 1 else 'Not Winner'}")

    # Confidence score if available
    if hasattr(model, 'predict_proba'):
        confidence = model.predict_proba(input_data)[0][1]
        st.info(f"🔍 Confidence Score: {confidence * 100:.2f}%")

    # Feedback animation
    if prediction == 1:
        st.balloons()
        st.markdown("🏆 This contestant has a high chance of winning!")
    else:
        st.markdown("📉 This contestant is less likely to win.")


