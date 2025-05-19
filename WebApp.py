import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load trained models and scaler
log_reg = joblib.load("models/log_reg_model.pkl")
knn_clf = joblib.load("models/knn_model.pkl")
rf_clf = joblib.load("models/rf_model.pkl")
xgb_clf = joblib.load("models/xgb_class_model.pkl")

# rf_reg = joblib.load("models/rf_reg_model.pkl")
# log_reg_reg = joblib.load("models/log_reg_reg_model.pkl")
# knn_reg = joblib.load("models/knn_reg_model.pkl")
# xgb_reg = joblib.load("models/xgb_reg_model.pkl")

scaler = joblib.load("scaler_class.pkl")

# Title
st.title("📱 Social Media Addiction Checker")
st.write("Answer the following questions to see if you're at risk of social media addiction.")

# User Inputs
age = st.slider("Your age", 10, 60, 20)
use_social_media = st.radio("Do you use social media?", ["Yes", "No"])
avg_social_time = st.selectbox("How much time do you spend on social media daily?", [
    "Less than an Hour",
    "Between 1 and 2 hours",
    "Between 2 and 3 hours",
    "Between 3 and 4 hours",
    "Between 4 and 5 hours",
    "More than 5 hours"
])

purposeless = st.slider("How often do you use social media without a purpose?", 1, 5, 3)
distracted = st.slider("How often does it distract you when you're busy?", 1, 5, 3)
restless = st.slider("Do you feel restless if you haven't used it in a while?", 1, 5, 3)
validation = st.slider("How often do you seek validation through social media?", 1, 5, 3)
comparison = st.slider("How often do you compare yourself to others online?", 1, 5, 3)
depressed = st.slider("How often do you feel depressed or down?", 1, 5, 3)
sleep_issues = st.slider("How often do you have sleep issues?", 1, 5, 3)

# Map categorical values
time_map = {
    "Less than an Hour": 1,
    "Between 1 and 2 hours": 2,
    "Between 2 and 3 hours": 3,
    "Between 3 and 4 hours": 4,
    "Between 4 and 5 hours": 5,
    "More than 5 hours": 6
}
use_map = {"Yes": 1, "No": 0}

avg_social_time_num = time_map[avg_social_time]
use_social_media_num = use_map[use_social_media]

# Assemble input
user_data = np.array([[age, use_social_media_num, avg_social_time_num,
                       purposeless, distracted, restless,
                       validation, comparison, depressed, sleep_issues]])

# Scale input
user_data_scaled = scaler.transform(user_data)

# Classification predictions
class_preds = [
    log_reg.predict(user_data_scaled)[0],
    knn_clf.predict(user_data_scaled)[0],
    rf_clf.predict(user_data_scaled)[0],
    xgb_clf.predict(user_data_scaled)[0]
]
addicted_votes = sum(class_preds)

# # Regression predictions
# reg_scores = [
#     rf_reg.predict(user_data_scaled)[0],
#     log_reg_reg.predict(user_data_scaled)[0],
#     knn_reg.predict(user_data_scaled)[0],
#     xgb_reg.predict(user_data_scaled)[0]
# ]
# avg_score = sum(reg_scores) / len(reg_scores)

# Results
st.subheader("🔍 Your Results")
st.write(f"{addicted_votes} of our 4 models think that you are addicted.")

if addicted_votes >= 3:
    st.error("You may be at high risk of social media addiction.")
    st.markdown("**Tips to improve:**")
    st.markdown("- Set screen time limits\n- Use focus apps to reduce distractions\n- Take regular breaks from social media\n- Spend more time on offline hobbies and relationships")
elif addicted_votes == 2:
    st.warning("Your results are mixed. Some signs of addiction may be present.")
    st.markdown("- Try tracking your screen time\n- Be mindful of how you feel during and after usage\n- Start setting small daily goals to reduce dependency")
elif addicted_votes == 1:
    st.success("Most models suggest you're likely not addicted, but stay aware.")
    st.markdown("- Continue healthy usage habits\n- Maintain boundaries for when and how you use social media")
else:
    st.success("You're likely not addicted to social media. Great job!")
    st.markdown("- Keep monitoring your habits\n- Share your healthy practices with friends!")

# st.metric("Your calculated addiction score is", f"{avg_score:.2f} / 10")