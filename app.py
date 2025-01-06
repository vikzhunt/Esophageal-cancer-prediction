import streamlit as st
import numpy as np
import joblib 
from sklearn.preprocessing import LabelEncoder


model = joblib.load('rf_model_cancer.pkl')

st.markdown("""
    <style>
    h1 {
        color: #629584;
        text-align: center;
        font-size: 3rem;
    }
    .sidebar .sidebar-content {
        padding-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Esophageal Cancer Status Prediction")

with st.sidebar:
    st.markdown("", unsafe_allow_html=True)
    days_to_birth = st.sidebar.slider("Days to Birth", min_value=-25000, max_value=0, value=-20000, step=1000)
    gender = st.sidebar.selectbox("Gender", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
    other_dx = st.sidebar.selectbox("Other Diagnosis", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    vital_status = st.sidebar.selectbox("Vital Status", options=[0, 1], format_func=lambda x: "Dead" if x == 0 else "Alive")
    has_new_tumor_events_information = st.sidebar.selectbox(
        "New Tumor Events Information", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes"
    )
    day_of_form_completion = st.sidebar.slider("Day of Form Completion", min_value=1, max_value=31, value=15, step=1)
    month_of_form_completion = st.sidebar.slider("Month of Form Completion", min_value=1, max_value=12, value=6, step=1)
    year_of_form_completion = st.sidebar.selectbox("Year of Form Completion", options=[2012, 2013, 2014, 2015])
    has_follow_ups_information = st.sidebar.selectbox(
        "Follow-Ups Information", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes"
    )
    has_drugs_information = st.sidebar.selectbox(
        "Drugs Information", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes"
    )
    has_radiations_information = st.sidebar.selectbox(
        "Radiations Information", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes"
    )
    stage_event_system_version = st.sidebar.selectbox(
        "Stage Event System Version", options=[0, 1, 2], format_func=lambda x: f"{x + 5}th"
    )
    primary_pathology_histological_type = st.sidebar.selectbox(
        "Primary Pathology Histological Type",
        options=[0, 1],
        format_func=lambda x: "Adenocarcinoma, NOS" if x == 0 else "Squamous Cell Carcinoma"
    )
    primary_pathology_neoplasm_histologic_grade = st.sidebar.selectbox(
        "Histologic Grade", options=[0, 1, 2, 3], format_func=lambda x: f"G{x + 1}" if x < 3 else "GX"
    )
    primary_pathology_age_at_initial_pathologic_diagnosis = st.sidebar.number_input(
        "Age at Initial Diagnosis (years)", min_value=30, max_value=100, value=50, step=1
    )


input_data = [
        days_to_birth,
        gender,
        other_dx,
        vital_status,
        has_new_tumor_events_information,
        day_of_form_completion,
        month_of_form_completion,
        year_of_form_completion,
        has_follow_ups_information,
        has_drugs_information,
        has_radiations_information,
        stage_event_system_version,
        primary_pathology_histological_type,
        primary_pathology_neoplasm_histologic_grade,
        primary_pathology_age_at_initial_pathologic_diagnosis
]

input = np.array(input_data);

if st.button("Predict"):
    pred = model.predict(input.reshape(1,-1))
    if pred[0] == 1:
        st.warning(f"The model predicts: Cancer Detected (Positive)")
    else:
        st.success(f"The model predicts: No Cancer (Negative)")

if st.checkbox("Show Feature Importance"):
    st.subheader("Feature Importance")
    importances = model.feature_importances_
    st.bar_chart(importances)