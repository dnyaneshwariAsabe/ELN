import streamlit as st
import pandas as pd
import pickle
import numpy as np
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Purchase Predictor AI",
    page_icon="🤖",
    layout="centered"
)

# --- Custom CSS for Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.02);
    }
    </style>
    """, unsafe_allow_html=True)

# --- Load Model ---
@st.cache_resource
def load_model():
    with open('model (4).pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# --- App Header ---
st.title("🤖 Purchase Prediction Dashboard")
st.markdown("Enter customer details below to predict the likelihood of a purchase.")
st.divider()

# --- Input Form ---
with st.container():
    col1, col2 = st.columns(2)
    
    with col1:
        user_id = st.number_input("User ID", min_value=0, value=15624510, step=1)
        gender = st.selectbox("Gender", ["Male", "Female"])
    
    with col2:
        age = st.slider("Age", 18, 100, 30)
        salary = st.number_input("Estimated Salary ($)", min_value=0, value=50000, step=500)

# --- Processing Input ---
# Map Gender to numerical if your model expects it (usually Male=1, Female=0 or vice versa)
# Note: Check if your training data used encoding. Assuming Male: 1, Female: 0
gender_encoded = 1 if gender == "Male" else 0

# Create DataFrame for prediction
input_data = pd.DataFrame([[user_id, gender_encoded, age, salary]], 
                          columns=['User ID', 'Gender', 'Age', 'EstimatedSalary'])

st.divider()

# --- Prediction Logic ---
if st.button("Analyze Customer Profile"):
    with st.spinner('Calculating probabilities...'):
        time.sleep(1) # Artificial delay for "Animated" feel
        
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)

        if prediction[0] == 1:
            st.balloons()
            st.success("### Result: High Likelihood of Purchase! ✅")
            st.metric(label="Confidence Score", value=f"{np.max(prediction_proba)*100:.2f}%")
        else:
            st.info("### Result: Unlikely to Purchase ❌")
            st.metric(label="Confidence Score", value=f"{np.max(prediction_proba)*100:.2f}%")

# --- Footer ---
st.caption("Powered by RandomForestClassifier | Machine Learning Deployment")
