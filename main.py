import streamlit as st
from keras.models import load_model
import numpy as np 
from sklearn.preprocessing import StandardScaler
import pandas as pd 

x_train = np.load("x_train.npy")

scaler = StandardScaler()
scaler.fit(x_train)
loaded_model = load_model("my_model.h5")

st.markdown('<h1 style="font-size: 30px;margin-top:-20px;">Healthcare & AI: CardioRiskNet - Predicting Coronary Heart Disease Risk with Neural Networks in the next 10 years</h1>', unsafe_allow_html=True)

def inputs():
    selected_gender = st.radio("Gender", ("Male", "Female"), key="gender_radio")
    if selected_gender == "Male":
        selected_gender = 1
    else:
        selected_gender = 0
    selected_age = st.slider("Age", 20, 80, 25)
    education = st.slider("Education", 1, 4, 1)
    currentSmoker = st.radio("Current Smoker", ("No", 'Yes'))
    if currentSmoker == "Yes":
        currentSmoker = 1
    else:
        currentSmoker = 0
    cgsPerday = st.slider("Cigarettes per day", 0, 70, 30)
    BPmeds = st.radio("Blood Pressure Medication", ("No", 'Yes'))
    if BPmeds == "Yes":
        BPmeds = 1
    else:
        BPmeds = 0
    prevalentStroke = st.radio("Prevalent Stroke", ("No", 'Yes'))
    if prevalentStroke == "Yes":
        prevalentStroke = 1
    else:
        prevalentStroke = 0
   
    
    prevalentHyp = st.radio("Prevalent Hypertension", ("No", 'Yes'))
    if prevalentHyp == "Yes":
        prevalentHyp = 1
    else:
        prevalentHyp = 0
    diabetes = st.radio("Diabetes", ("No", 'Yes'))
    if diabetes == "Yes":
        diabetes = 1
    else:
        diabetes = 0
    totChol = st.slider("Total Cholesterol", 100, 700, 200)
    sysBp = st.slider("Systolic Blood Pressure", 80, 300, 120)
    diaBp = st.slider("Diastolic Blood Pressure", 40, 200, 80)
    bmi = st.slider("Body Mass Index", 10, 70, 20)
    heartrate = st.slider("Heart Rate", 40, 150, 80)
    glucose = st.slider("Glucose", 40, 400, 100)
    
    
    
    return selected_gender, selected_age,education,currentSmoker, cgsPerday, BPmeds, prevalentStroke, prevalentHyp, diabetes, totChol, sysBp, diaBp, bmi, heartrate, glucose
if __name__ == "__main__":
    input_values = np.array(list(inputs())).reshape(1, -1)
      # Reshape to (1, num_features)
    x_train_scaled = scaler.transform(input_values)
    if st.button("Submit"):
        prediction = loaded_model.predict(x_train_scaled)
        risk_percentage = round(float(prediction[0][0] * 100), 2)
        if risk_percentage >= 50:
            st.write(f"High Risk: Predicted Risk is {risk_percentage}%")
            st.warning("You have a high risk of coronary heart disease. Please consult a healthcare professional.")
        else:
            st.write(f"Low Risk: Predicted Risk is {risk_percentage}%")
            st.success("Your risk of coronary heart disease is relatively low. Keep up the good work!")