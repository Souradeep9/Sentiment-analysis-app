import streamlit as st
import joblib

# Load saved model
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("Sentiment Analysis App")

text = st.text_input("Enter text")

if st.button("Predict"):
    vec = vectorizer.transform([text])
    result = model.predict(vec)

    if result[0] == 1:
        st.write("Positive 😊")
    else:
        st.write("Negative 😡")