

import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('winner_prediction_model.pkl')

# Full Profession List
profession_list = [
    'Soap Actor', 'Model', 'Pageant Winner', 'Activist', 'Actor',
    'Actress', 'Commoner', 'Soap Actress', 'Sports Person', 'Singer',
    'Comedian', 'Reality Show Alumni', 'International Star',
    'Politician', 'Film Director', 'Fashion Designer', 'Lawyer',
    'TV Anchor', 'Controversial Fame', 'Wrestler', 'Choreographer',
    'News Presenter', 'Cartoonist', 'Celebrity Hairstylist',
    'Producer', 'Video Jockey', 'Radio Jockey', 'Commoner - Teacher',
    'Stage Actress', 'Commoner - Student', 'Commoner - Farmer',
    'Commoner - Employee', 'Commoner - Marketing', 'Swamiji',
    'Commoner - IT Professional', 'Commoner - Engineer',
    'Commoner - Housewife', 'Dancer', 'Social Media Star',
    'Commoner - Police', 'Commoner - Lawyer', 'Commoner - Entrepreneur',
    'Commoner - Receptionist', 'Entrepreneur', 'Commoner - Sales Manager',
    'Commoner - Singer', 'Commoner - Doctor', 'Writer', 'Political Analyst',
    'Musician', 'Doctor', 'Film director', 'Theatre Artist', 'Pharmacist',
    'Businessman', 'Astrologer', 'Boxer', 'Art Director', 'Air Hostess',
    'Commoner - Sales Representative', 'Numerologist', 'Commoner - Priest',
    'Commoner - Bus Conductor', 'Commoner - Junior Artist',
    'Commoner - Dubbing Artist', 'Journalist', 'Farmer', 'Snake enthusiast',
    'Film Critic', 'Commoner - RJ', 'Commoner - Social Activist',
    'Commoner - Pageant Winner', 'Lyricist', 'Commoner - Voice Trainer',
    'Folk Artist', 'Chef', 'Preacher', 'Gym Trainer', 'Author',
    'International star', 'Dubbing Artist', 'Actor & Actress', 'Magician',
    'Photograher', 'Fitness Trainer', 'Makeup artist', 'Commoner - Adventurer',
    'Disc Jockey'
]
profession_map = {prof: idx for idx, prof in enumerate(profession_list)}

# Most Viewed State List
states_list = [
    'Maharashtra, Bihar, Delhi, Haryana, Jharkhand, Madhya Pradesh, Rajasthan, Uttarakhand, Uttar Pradesh',
    'Karnataka',
    'Telangana, Andhra Pradesh',
    'Tamil Nadu',
    'Maharashtra',
    'West Bengal',
    'Kerala'
]
states_map = {state: idx for idx, state in enumerate(states_list)}

# Gender Categories
gender_list = ['Male', 'Female', 'Transgender', 'Pair']
gender_map = {gender: idx for idx, gender in enumerate(gender_list)}

# Updated House Location Categories
house_locations = ['Lonavala', 'Karjat', 'Mumbai', 'Bengaluru', 'Hyderabad', 'Chennai']
house_map = {loc: idx for idx, loc in enumerate(house_locations)}

# Age Category Mapping
age_category_list = ['Young', 'Middle', 'Old']
age_map = {'Young': 0, 'Middle': 1, 'Old': 2}

# Other mappings
wildcard_map = {'Yes': 1, 'No': 0}
ott_map = {'Yes': 1, 'No': 0}
finalist_map = {'Yes': 1, 'No': 0}

# Streamlit UI
st.set_page_config(page_title="Bigg Boss Winner Prediction", page_icon="🏆")
st.title("🏆 Bigg Boss Winner Prediction App")

st.markdown("""
Welcome to the **Bigg Boss Winner Prediction App**!  
Fill in the contestant's details to predict their chances of winning the show.
""")

# Input Fields
profession = st.selectbox("Profession", list(profession_map.keys()))
gender = st.selectbox("Gender", gender_list)
age_category = st.selectbox("Age Category", age_category_list)
wild_card = st.selectbox("Wild Card Entry", list(wildcard_map.keys()))
most_viewed_state = st.selectbox("Most Viewed State", list(states_map.keys()))
house_location = st.selectbox("House Location", house_locations)
ott_season = st.selectbox("OTT Season", list(ott_map.keys()))
num_evictions = st.slider("Number of Evictions Faced", 0, 10, 1)
finalist = st.selectbox("Is Finalist?", list(finalist_map.keys()))

# Create input data
input_data = pd.DataFrame([{
    'Profession': profession_map[profession],
    'Gender': gender_map[gender],
    'Age': age_map[age_category],
    'Wild Card': wildcard_map[wild_card],
    'Most Viewed States': states_map[most_viewed_state],
    'House Location': house_map[house_location],
    'OTT Season': ott_map[ott_season],
    'Number of Evictions Faced': num_evictions,
    'Finalist': finalist_map[finalist]
}])

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    st.success(f"🎯 Prediction: {'Winner' if prediction == 1 else 'Not Winner'}")

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







