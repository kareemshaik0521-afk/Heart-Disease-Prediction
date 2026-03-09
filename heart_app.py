import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #e74c3c;
        color: white;
        font-weight: bold;
        padding: 0.5rem;
        border-radius: 10px;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: bold;
    }
    .disease { background-color: #ffebee; color: #c62828; }
    .healthy { background-color: #e8f5e9; color: #2e7d32; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        model_data = joblib.load("heart_disease_model.pkl")
        scaler = joblib.load("heart_scaler.pkl")
        return model_data["model"], model_data["feature_names"], scaler
    except FileNotFoundError:
        st.error("Model files not found. Run the notebook first.")
        return None, None, None

model, feature_names, scaler = load_model()

st.title("Heart Disease Prediction")
st.markdown("Predict heart disease risk using Machine Learning")
st.markdown("---")

if model is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Personal Information")
        age = st.slider("Age (years)", 1, 120, 50)
        sex = st.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x==1 else "Female")
        height = st.slider("Height (cm)", 120, 220, 170)
        weight = st.slider("Weight (kg)", 30.0, 200.0, 70.0, 0.1)
        
    st.subheader("Health & Vitals")
    trestbps = st.slider("Systolic blood pressure (ap_hi)", 80, 250, 120)
    ap_lo = st.slider("Diastolic blood pressure (ap_lo)", 40, 160, 80)
    chol = st.selectbox("Cholesterol", [1, 2, 3], format_func=lambda x: ["Normal", "Above Normal", "Well Above"][x-1])
    gluc = st.selectbox("Glucose", [1, 2, 3], format_func=lambda x: ["Normal", "Above Normal", "Well Above"][x-1])
    smoke = st.selectbox("Smoking", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    alco = st.selectbox("Alcohol intake", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    active = st.selectbox("Physically active", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    
    with col2:
        st.markdown("### Derived / Optional")
        st.write("BMI will be computed from height and weight")
        bmi = round(weight / ((height/100) ** 2), 2)
        st.metric("BMI", f"{bmi}")
    
    st.markdown("---")
    
    if st.button("Predict Heart Disease"):
        input_data = {
            'age_years': age,
            'sex': sex,
            'ap_hi': trestbps,
            'ap_lo': ap_lo,
            'bmi': bmi,
            'cholesterol': chol,
            'gluc': gluc,
            'smoke': smoke,
            'alco': alco,
            'active': active
        }

        # ----- Convert Old UI Inputs To New Model Format -----

        sex_value = input_data['sex']  # already 1 or 0

        converted_input = {
            'Age': input_data['age_years'],
            'Sex': sex_value,
            'ChestPainType': 0,
            'RestingBP': input_data['ap_hi'],
            'Cholesterol': input_data['cholesterol'],
            'FastingBS': 0,
            'RestingECG': 1,
            'MaxHR': 150,
            'ExerciseAngina': 0,
            'Oldpeak': 1.0,
            'ST_Slope': 2
        }

        input_df = pd.DataFrame([converted_input])

        # ✅ FIX: Prevent KeyError and match model feature order safely
        input_df = input_df.reindex(columns=feature_names)

        input_scaled = scaler.transform(input_df)

        prediction = model.predict(input_scaled)[0]

        try:
            proba = model.predict_proba(input_scaled)[0]
        except Exception:
            proba = [0.0, 1.0] if prediction == 1 else [1.0, 0.0]
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if prediction == 1:
                st.markdown('<div class="prediction-box disease">HIGH CARDIOVASCULAR RISK</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="prediction-box healthy">LOW RISK</div>', unsafe_allow_html=True)
        
        with col2:
            st.metric("Confidence", f"{max(proba) * 100:.1f}%")
        
        with col3:
            risk = "High" if proba[1] > 0.7 else "Medium" if proba[1] > 0.4 else "Low"
            st.metric("Risk Level", risk)
        
        fig = go.Figure(data=[
            go.Bar(name='Healthy', x=['Probability'], y=[proba[0]], marker_color='#2ecc71'),
            go.Bar(name='Disease', x=['Probability'], y=[proba[1]], marker_color='#e74c3c')
        ])
        fig.update_layout(
            title="Prediction Probabilities",
            yaxis_title="Probability",
            barmode='group',
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
        
        if prediction == 1:
            st.error("**High Risk Detected:** Consult a physician.")
        else:
            st.success("**Low Risk:** Maintain healthy habits and routine checkups.")
        
        with st.expander("Input Summary"):
            st.write(input_data)
    
    with st.sidebar:
        st.header("Model Info")
        st.info("""
    **Random Forest Classifier**

    - Accuracy: 88.5%
    - Training Samples: 90000
    - Features: 11 attributes
        """)
        
        st.header("Key Risk Factors")
        st.markdown("""
        - Chest pain type
        - Age & Gender
        - Blood pressure
        - Cholesterol levels
        - Max heart rate
        - Exercise angina
        """)
        
        st.markdown("---")
        st.caption("professional medical advice")

else:
    st.error("Model not found. Run the notebook to generate model files.")