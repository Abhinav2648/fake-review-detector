
import streamlit as st
import pickle

st.title("🛍️ Fake Review Detection")

model = pickle.load(open("fake_review_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

review = st.text_area("Enter your review:")

if st.button("Predict"):
    if review.strip() == "":
        st.warning("Please enter a review")
    else:
        vec = vectorizer.transform([review])
        result = model.predict(vec)

        if result[0] == 1:
            st.error("Fake Review ❌")
        else:
            st.success("Genuine Review ✅")
