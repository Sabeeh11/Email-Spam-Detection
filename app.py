import streamlit as st
import joblib

model = joblib.load("models/spam_model.pkl")

st.title("Email Spam Detection App")
st.write("Enter an email or SMS message to classify it as Spam or Not Spam.")

message = st.text_area("Message")

if st.button("Predict"):
    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        prediction = model.predict([message])[0]
        probability = model.predict_proba([message])[0]

        if prediction == 1:
            st.error(f"Spam detected. Confidence: {probability[1]:.2%}")
        else:
            st.success(f"Not spam. Confidence: {probability[0]:.2%}")
