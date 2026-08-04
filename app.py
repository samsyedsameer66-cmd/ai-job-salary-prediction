import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("model.pkl")

st.set_page_config(
    page_title="AI Job Salary Prediction",
    page_icon="💼",
    layout="centered"
)

st.title("💼 AI Job Salary Prediction App")

st.write("Enter the job details below:")

JOB_TITLES = [
    "AI Research Scientist", "AI Software Engineer", "AI Specialist", "NLP Engineer",
    "AI Consultant", "AI Architect", "Principal Data Scientist", "Data Analyst",
    "Machine Learning Engineer", "Data Scientist", "Data Engineer", "ML Ops Engineer",
    "Computer Vision Engineer", "Deep Learning Engineer", "AI Product Manager"
]
EXPERIENCE_LEVELS = {"EN": "Entry-level", "MI": "Mid-level", "SE": "Senior", "EX": "Executive"}
EMPLOYMENT_TYPES = {"FT": "Full-time", "PT": "Part-time", "CT": "Contract", "FL": "Freelance"}
COMPANY_SIZES = {"S": "Small", "M": "Medium", "L": "Large"}
EDUCATION_LEVELS = ["Associate", "Bachelor", "Master", "PhD"]
COUNTRIES = [
    "United States", "United Kingdom", "Canada", "Germany", "France", "India",
    "China", "Ireland", "South Korea", "Singapore", "Switzerland", "Australia",
    "Japan", "Netherlands", "Brazil", "Sweden", "Israel", "Norway", "Denmark", "Finland"
]
INDUSTRIES = [
    "Automotive", "Media", "Education", "Consulting", "Healthcare", "Gaming",
    "Government", "Telecommunications", "Finance", "Retail", "Technology",
    "Manufacturing", "Energy", "Real Estate", "Transportation"
]

# Inputs
job_title = st.selectbox("Job Title", JOB_TITLES)

col1, col2 = st.columns(2)
with col1:
    experience_level = st.selectbox(
        "Experience Level", list(EXPERIENCE_LEVELS.keys()),
        format_func=lambda x: EXPERIENCE_LEVELS[x]
    )
    employment_type = st.selectbox(
        "Employment Type", list(EMPLOYMENT_TYPES.keys()),
        format_func=lambda x: EMPLOYMENT_TYPES[x]
    )
    company_size = st.selectbox(
        "Company Size", list(COMPANY_SIZES.keys()),
        format_func=lambda x: COMPANY_SIZES[x]
    )
    education_required = st.selectbox("Education Required", EDUCATION_LEVELS)

with col2:
    company_location = st.selectbox("Company Location", COUNTRIES)
    employee_residence = st.selectbox("Employee Residence", COUNTRIES)
    industry = st.selectbox("Industry", INDUSTRIES)
    remote_ratio = st.select_slider("Remote Ratio (%)", options=[0, 50, 100], value=50)

years_experience = st.number_input("Years of Experience", min_value=0, max_value=40, value=3)
num_skills = st.slider("Number of Required Skills", min_value=1, max_value=10, value=4)
job_description_length = st.number_input(
    "Job Description Length (characters)", min_value=0, value=1200
)
benefits_score = st.slider("Benefits Score", min_value=0.0, max_value=10.0, value=6.0, step=0.1)
days_to_deadline = st.number_input(
    "Days Until Application Deadline", min_value=1, max_value=120, value=30
)

if st.button("Predict Salary"):

    input_data = pd.DataFrame({
        "remote_ratio": [remote_ratio],
        "years_experience": [years_experience],
        "job_description_length": [job_description_length],
        "benefits_score": [benefits_score],
        "num_skills": [num_skills],
        "days_to_deadline": [days_to_deadline],
        "job_title": [job_title],
        "experience_level": [experience_level],
        "employment_type": [employment_type],
        "company_location": [company_location],
        "company_size": [company_size],
        "employee_residence": [employee_residence],
        "education_required": [education_required],
        "industry": [industry],
    })

    prediction = model.predict(input_data)[0]

    st.success(f"Predicted Salary: ${prediction:,.0f} USD")
