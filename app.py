import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("model/iris_model.pkl")

st.title("🌸 Iris Flower Prediction Web App")
st.write("Enter flower measurements to predict the species")

# Input fields
sepal_length = st.number_input("Sepal Length", min_value=0.0, max_value=10.0, value=5.1)
sepal_width = st.number_input("Sepal Width", min_value=0.0, max_value=10.0, value=3.5)
petal_length = st.number_input("Petal Length", min_value=0.0, max_value=10.0, value=1.4)
petal_width = st.number_input("Petal Width", min_value=0.0, max_value=10.0, value=0.2)

if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("🌼 Prediction: Iris Versicolor")
    else:
        st.success("🌺 Prediction: Iris Virginica")
