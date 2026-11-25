
import streamlit as st
import joblib
import pandas as pd

# Load trained AI model
model = joblib.load("R15_pleural_effusion_model.pkl")

# Application settings
st.set_page_config(
    page_title="Pleural AI",
    page_icon="🫁",
    layout="centered"
)

st.title("🫁 Pleural AI")
st.subheader("AI-Powered Pleural Effusion Report Classifier")

st.write(
    "Enter a radiology report below to classify "
    "its pleural effusion finding."
)

st.info(
    "Educational prototype using synthetic text data. "
    "Not intended for clinical diagnosis."
)

# User input
report = st.text_area(
    "Enter radiology report:",
    placeholder="Example: There is evidence of bilateral pleural effusion.",
    height=150
)

# Prediction
if st.button("Analyze Report", type="primary"):

    if report.strip():

        prediction = model.predict([report])[0]

        st.success("Analysis completed!")

        st.subheader("AI Prediction")

        if prediction == "Present":
            st.warning("Pleural Effusion: PRESENT")

        elif prediction == "Absent":
            st.success("Pleural Effusion: ABSENT")

        else:
            st.info("Pleural Effusion: UNCERTAIN")

        # Confidence scores
        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba([report])[0]

            st.subheader("Prediction Confidence")

            results = pd.DataFrame({
                "Finding": model.classes_,
                "Confidence (%)": probabilities * 100
            })

            results["Confidence (%)"] = results[
                "Confidence (%)"
            ].round(2)

            st.dataframe(results, hide_index=True)

            st.bar_chart(
                results.set_index("Finding")
                ["Confidence (%)"]
            )

    else:
        st.error("Please enter a radiology report.")

st.divider()

st.caption("HCT487 | R15 | AI in Healthcare")
st.caption("Developed for educational purposes.")
