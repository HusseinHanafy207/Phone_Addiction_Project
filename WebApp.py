import streamlit as st
import numpy as np
import pandas as pd
import joblib
from PIL import Image

# Load trained classification models and scaler
log_reg = joblib.load("models/log_reg_model.pkl")
knn_clf = joblib.load("models/knn_model.pkl")
rf_clf = joblib.load("models/rf_model.pkl")
xgb_clf = joblib.load("models/xgb_class_model.pkl")
scaler = joblib.load("scaler_class.pkl")

# Feature names in training order
feature_names = [
    'age','use_social_media','avg_social_time','purposeless_use_freq',
    'distracted_by_sm','restless_without_sm','easily_distracted',
    'worry_level','concentration_difficulty','comparison_freq',
    'comparison_feelings','seek_validation','feel_depressed',
    'interest_fluctuation','sleep_issues',
    'gender_Female','gender_Male',
    'relationship_status_In a relationship',
    'relationship_status_Married',
    'relationship_status_Single'
]

st.title("📱 Social Media Addiction Checker")
st.write("Answer the questions below to assess your risk of social media addiction.")

# --- User Inputs ---
age = st.slider("Your age", 10, 60, 25)
use_sm = st.radio("Do you use social media?", ["Yes", "No"])
avg_time = st.selectbox(
    "Average daily social media use:",
    [
        "Less than an Hour",
        "Between 1 and 2 hours",
        "Between 2 and 3 hours",
        "Between 3 and 4 hours",
        "Between 4 and 5 hours",
        "More than 5 hours"
    ]
)
# Behavioral sliders
purposeless = st.slider("Use social media without purpose (1=Never,5=Always)", 1, 5, 3)
distracted = st.slider("Distraction by social media (1=Never,5=Always)", 1, 5, 3)
restless = st.slider("Restlessness if not using social media (1=Never,5=Always)", 1, 5, 3)
easily_distracted = st.slider("Ease of distraction (1=Low,5=High)", 1, 5, 3)
worry_level = st.slider("Worry level (1=Low,5=High)", 1, 5, 3)
concentration = st.slider("Difficulty concentrating (1=Low,5=High)", 1, 5, 3)
comparison = st.slider("Comparison to other people on social media frequency (1=Never,5=Always)", 1, 5, 3)
comparison_feelings = st.slider("Feelings after comparison (1=Poorly affected,5=Not affected)", 1, 5, 3)
validation = st.slider("Validation seeking (1=Never,5=Always)", 1, 5, 3)
depressed = st.slider("Depression frequency (1=Never,5=Always)", 1, 5, 3)
interest = st.slider("Interest fluctuation in things that you usually like to do (1=Low,5=High)", 1, 5, 3)
sleep_issues = st.slider("Sleep issues (1=Never,5=Always)", 1, 5, 3)

# Demographics
gender = st.radio("Gender", ["Male", "Female"])
rel_status = st.selectbox(
    "Relationship status:",
    ["Single", "In a relationship", "Married"]
)

# --- Encoding & Preprocessing ---
time_map = {
    "Less than an Hour": 1,
    "Between 1 and 2 hours": 2,
    "Between 2 and 3 hours": 3,
    "Between 3 and 4 hours": 4,
    "Between 4 and 5 hours": 5,
    "More than 5 hours": 6
}
use_map = {"Yes": 1, "No": 0}
gender_map = {"Female": (1, 0), "Male": (0, 1)}  # female, male
rel_map = {
    "Single": (1, 0, 0),
    "In a relationship": (0, 1, 0),
    "Married": (0, 0, 1)
}

# Map inputs
time_num = time_map[avg_time]
use_num = use_map[use_sm]
f_flag, m_flag = gender_map[gender]
s_flag, rel_flag, mar_flag = rel_map[rel_status]

# Assemble features
features = [
    age, use_num, time_num, purposeless, distracted, restless,
    easily_distracted, worry_level, concentration, comparison,
    comparison_feelings, validation, depressed, interest, sleep_issues,
    f_flag, m_flag, rel_flag, mar_flag, s_flag
]

# Prepare DataFrame and scale
X_input = pd.DataFrame([features], columns=feature_names)
X_scaled = scaler.transform(X_input)

# Show results on button click
if st.button("Get Results"):
    # Classification votes
    preds = [
        log_reg.predict(X_scaled)[0],
        knn_clf.predict(X_scaled)[0],
        rf_clf.predict(X_scaled)[0],
        xgb_clf.predict(X_scaled)[0]
    ]
    votes = sum(preds)

    st.subheader("🔍 Your Results")
    st.write(f"**{votes} of our 4 models** predict you are addicted.")

    # Verdict messages
    if votes >= 3:
        st.error("High risk of social media addiction.")
    elif votes == 2:
        st.warning("Mixed signals: moderate risk detected.")
    elif votes == 1:
        st.info("Low risk, but keep an eye on your habits.")
    else:
        st.success("No signs of addiction.")

    # Enhanced insights and resources
    if votes >= 3:
        st.markdown("**Tips to reduce addiction:**")
        st.markdown(
            "- Set daily screen time limits and schedule phone-free hours.\n"
            "- Use focus apps (e.g., Forest, Freedom) to block distracting notifications.\n"
            "- Engage in offline activities: exercise, reading, or hobbies.\n"
            "- Practice mindfulness and take digital detox days."
        )
        st.markdown("**Did you know?** According to a 2021 Pew Research study, adults spend an average of 2 hours 24 minutes per day on social media. Overuse is linked to increased anxiety and depression (APA, 2019).")
        st.markdown("**Learn more:**")
        st.markdown(
            "- [Pew Research: Social Media Use in 2021](https://www.pewresearch.org/internet/2021/04/07/social-media-use-in-2021/)\n"
            "- [Digital Detox Guide by NYTimes](https://www.nytimes.com/2024/01/12/well/live/tech-digital-detox-screen-time.html)\n"
            "- [APA: Internet and Technology](https://www.apa.org/topics/social-media-internet)"
        )
    elif votes == 2:
        st.markdown("**Moderate risk insights:**")
        st.markdown(
            "- Track your screen time using built-in phone features.\n"
            "- Set small, achievable goals to reduce daily social media use by 15 minutes.\n"
            "- Reflect on your feelings: keep a usage journal."
        )
        st.markdown("**Statistics:** In a survey, 48% of users reported feeling more anxious after prolonged social media sessions (JAMA Pediatrics, 2020).")
        st.markdown("**Read:** [Mindful Social Media Use](https://link.springer.com/article/10.1007/s12671-023-02271-9)")
    else:
        st.markdown("**Great balance!** You’re using social media mindfully.")
        st.markdown(
            "- Continue monitoring your habits.\n"
            "- Share healthy usage practices with friends and family.\n"
            "- Explore positive content: educational and creative communities."
        )
        st.markdown("**Stats:** Top 10% of balanced users report 30% higher life satisfaction (Global Web Index, 2020).")

# Footer: Logo and Credits
logo = Image.open('euilogo.png')
st.image(logo, width=200)
st.markdown("---")
st.markdown("**Created by:**")
st.markdown(
    "- Ali Ayman (22-101190)\n"
    "- Hussein Hanafy (22-101184)\n"
    "- Mohamed Ayman (22-101182)\n"
    "- Fares Gamil (22-101100)\n"
    "- Mohamed Atta (22-101187)\n"
    "- Kareem Yasser (22-101124)"
)