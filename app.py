import streamlit as st
import joblib
import re


def clean_text(text):
    text = re.sub(r"[^a-zA-Z]", " ", str(text))
    text = text.lower()
    text = text.strip()
    return text


model = joblib.load("spam_model.pkl")

st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="📧",
    layout="centered"
)

st.title("Email Spam Detection App")
st.write(
    "This app uses Natural Language Processing and Machine Learning "
    "to classify emails as Spam or Ham."
)

message = st.text_area("Enter email text:")

if st.button("Predict"):
    if message.strip() == "":
        st.warning("Please enter an email message.")
    else:
        cleaned_message = clean_text(message)

        prediction = model.predict([cleaned_message])[0]
        probability = model.predict_proba([cleaned_message])[0]

        if prediction == 1:
            st.error(f"Spam detected with {probability[1] * 100:.2f}% confidence.")
        else:
            st.success(f"Ham detected with {probability[0] * 100:.2f}% confidence.")

st.markdown("---")
st.caption("Model: TF-IDF + Logistic Regression")
