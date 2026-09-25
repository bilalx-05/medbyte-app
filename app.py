import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load data and train model
url = "pima-indians-diabetes.data.csv"
cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv(url, names=cols)
X = df.drop('Outcome', axis=1)
y = df['Outcome']
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Build the Web UI
st.title("🩺 MEDBYTE: AI Disease Prediction")
st.write("Adjust the patient details below to predict Diabetes risk.")

col1, col2 = st.columns(2)
with col1:
    preg = st.slider("Pregnancies", 0, 20, 1)
    glucose = st.slider("Glucose Level", 0, 200, 100)
    bp = st.slider("Blood Pressure", 0, 150, 70)
    skin = st.slider("Skin Thickness", 0, 100, 20)
with col2:
    insulin = st.slider("Insulin", 0, 900, 80)
    bmi = st.slider("BMI", 0.0, 70.0, 25.0)
    dpf = st.slider("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    age = st.slider("Age", 1, 100, 30)

if st.button("Predict"):
    input_data = [[preg, glucose, bp, skin, insulin, bmi, dpf, age]]
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.error("⚠️ High Risk of Diabetes Detected")
    else:
        st.success("✅ Low Risk of Diabetes")