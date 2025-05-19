import streamlit as st
import numpy as np
import pandas as pd
import joblib

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
distracted = st.slider("Distraction frequency (1=Never,5=Always)", 1, 5, 3)
restless = st.slider("Restlessness if not using social media (1=Never,5=Always)", 1, 5, 3)
easily_distracted = st.slider("Ease of distraction (1=Low,5=High)", 1, 5, 3)
worry_level = st.slider("Worry level (1=Low,5=High)", 1, 5, 3)
concentration = st.slider("Difficulty concentrating (1=Low,5=High)", 1, 5, 3)
comparison = st.slider("Comparison frequency (1=Never,5=Always)", 1, 5, 3)
comparison_feelings = st.slider("Feelings after comparison (1=Poorly affected,5=Not affected)", 1, 5, 3)
validation = st.slider("Validation seeking (1=Never,5=Always)", 1, 5, 3)
depressed = st.slider("Depression frequency (1=Never,5=Always)", 1, 5, 3)
interest = st.slider("Interest fluctuation (1=Low,5=High)", 1, 5, 3)
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

    if votes >= 3:
        st.error("High risk of social media addiction.")
    elif votes == 2:
        st.warning("Mixed signals: moderate risk detected.")
    elif votes == 1:
        st.info("Low risk, but keep an eye on your habits.")
    else:
        st.success("No signs of addiction.")

    # Insights
    if votes >= 3:
        st.markdown("**Tips to reduce addiction:**")
        st.markdown(
            "- Set daily screen time limits\n"
            "- Use focus apps to block notifications\n"
            "- Schedule offline activities (exercise, reading)\n"
            "- Practice mindfulness when using social media"
        )
    elif votes == 2:
        st.markdown("Consider tracking and moderating your usage to avoid escalation.")
    else:
        st.markdown("Great balance! Continue healthy social media habits.")