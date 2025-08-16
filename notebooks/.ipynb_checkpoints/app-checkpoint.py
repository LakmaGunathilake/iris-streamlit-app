import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load('data/best_model.pkl')

# App title
st.title("Iris Flower Species Predictor")

# Input fields
sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.8)
sepal_width = st.slider("Sepal Width", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal Length", 1.0, 7.0, 4.3)
petal_width = st.slider("Petal Width", 0.1, 2.5, 1.3)

# Predict button
if st.button("Predict"):
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(input_data)[0]
    species = ['setosa', 'versicolor', 'virginica'][prediction]
    st.success(f"The predicted species is: {species}")