from PIL import Image
import streamlit as st
import streamlit as st
import numpy as np
import joblib
from PIL import Image

# Load the trained model and scaler
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

# Custom Streamlit Page Configuration
st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="wide")

# Sidebar Section
with st.sidebar:
    st.title("🔍 About This App")
    st.write("This AI-powered app predicts the likelihood of diabetes based on medical inputs.")
    st.write("Developed using **Machine Learning** and **Streamlit** for an engaging experience.")
    st.markdown("---")
    st.write("💡 **Tip:** Enter valid medical values to get accurate predictions.")

# Main Title
st.markdown(
    """
    <h1 style='text-align: center; color: #2E86C1;'> Diabetes Detection </h1>
    <p style='text-align: center; color: #A93226;'> Enter your medical details and get an instant prediction! </p>
    """,
    unsafe_allow_html=True,
)


# Create two columns (Left: Image, Right: Form)
col1, col2 = st.columns([1, 2])  # Adjust ratio to control space distribution

with col1:
    # Load and display the image
    image = Image.open("diabetes.jpg")  # Ensure the image is in the same directory as app.py
    st.image(image, use_container_width=True)# Use full column width for better alignment

with col2:
    st.subheader("🔢 Enter Your Medical Details")
    pregnancies = st.slider("Number of Pregnancies", 0, 20, 1)
    glucose = st.slider("Glucose Level (mg/dL)", 0, 300, 120)
    blood_pressure = st.slider("Blood Pressure (mmHg)", 0, 200, 70)
    skin_thickness = st.slider("Skin Thickness (mm)", 0, 100, 20)
    insulin = st.slider("Insulin Level (mu U/ml)", 0, 900, 85)
    bmi = st.slider("BMI", 0.0, 70.0, 25.0)
    diabetes_pedigree = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
    age = st.slider("Age", 1, 120, 30)


# Predict Button with Animation
if st.button("🔍 Predict Now", use_container_width=True):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    
    if prediction[0] == 1:
        st.error(" High Risk! You may have Diabetes.")
        st.image("https://cdn.pixabay.com/photo/2018/06/18/20/55/doctor-3489633_1280.jpg", use_container_width=True)
    else:
        st.success(" Low Risk! You are not diabetic.")
        st.image("https://cdn.pixabay.com/photo/2016/03/09/15/30/medic-1240510_1280.jpg", use_container_width=True)

# Footer
st.markdown(
    """
    <hr>
    <p style='text-align: center;'> Developed by Shlok, Srushti ,Prathamesh  </p>
    """,
    unsafe_allow_html=True,
)
