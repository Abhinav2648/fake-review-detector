import streamlit as st
import joblib

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("Fake Review Detection")

review = st.text_area("Enter your review:")

if st.button("Predict"):
    vec = vectorizer.transform([review])
    result = model.predict(vec)[0]
    
    if result == 1:
        st.error("Fake Review ❌")
    else:
        st.success("Real Review ✅")

