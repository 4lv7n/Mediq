import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap

# -----------------------------
# Load Model and Files
# -----------------------------
model = joblib.load("mediq_model.pkl")
le = joblib.load("label_encoder.pkl")
kb = pd.read_excel("MEDIQ_Disease_Knowledge_Base.xlsx")

# -----------------------------
# Page Settings
# -----------------------------
st.set_page_config(
    page_title="MEDIQ",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 MEDIQ")
st.subheader("Explainable AI Clinical Decision Support System")

st.markdown("---")

# -----------------------------
# Patient Inputs
# -----------------------------
st.header("Patient Clinical Data")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=100, value=30)
    gender = st.selectbox("Gender", [0, 1])

    esr = st.number_input("ESR", value=20.0)
    crp = st.number_input("CRP", value=5.0)
    rf = st.number_input("RF", value=10.0)

    anti_ccp = st.selectbox("Anti-CCP", [0, 1])
    hla_b27 = st.selectbox("HLA-B27", [0, 1])

with col2:
    ana = st.selectbox("ANA", [0, 1])

    anti_ro = st.selectbox("Anti-Ro", [0, 1])
    anti_la = st.selectbox("Anti-La", [0, 1])

    anti_dsDNA = st.selectbox("Anti-dsDNA", [0, 1])
    anti_sm = st.selectbox("Anti-Sm", [0, 1])

    c3 = st.number_input("C3", value=100.0)
    c4 = st.number_input("C4", value=20.0)

# -----------------------------
# Predict Button
# -----------------------------
if st.button("Predict Disease"):

    patient = np.array([[
        age,
        gender,
        esr,
        crp,
        rf,
        anti_ccp,
        hla_b27,
        ana,
        anti_ro,
        anti_la,
        anti_dsDNA,
        anti_sm,
        c3,
        c4
    ]])

    try:

        # Prediction
        prediction = model.predict(patient)[0]
        disease = le.inverse_transform([prediction])[0]

        confidence = np.max(
            model.predict_proba(patient)
        ) * 100

        st.success(f"Predicted Disease: {disease}")
        st.info(f"Confidence: {confidence:.2f}%")        # -----------------------------
        # SHAP Explainability
        # -----------------------------
        try:

            feature_names = [
                "Age",
                "Gender",
                "ESR",
                "CRP",
                "RF",
                "Anti-CCP",
                "HLA-B27",
                "ANA",
                "Anti-Ro",
                "Anti-La",
                "Anti-dsDNA",
                "Anti-Sm",
                "C3",
                "C4"
            ]

            explainer = shap.TreeExplainer(model)

            shap_values = explainer.shap_values(patient)

            # Your SHAP shape = (1,14,7)
            importance = np.abs(
                shap_values[0, :, prediction]
            )

            shap_df = pd.DataFrame({
                "Biomarker": feature_names,
                "Impact": importance
            })

            top_features = (
                shap_df
                .sort_values(
                    by="Impact",
                    ascending=False
                )
                .head(5)
            )

            st.markdown("---")
            st.subheader("🔍 Top Contributing Biomarkers")

            for _, row_shap in top_features.iterrows():

                st.write(
                    f"• {row_shap['Biomarker']} → {row_shap['Impact']:.3f}"
                )

        except Exception as shap_error:

            st.warning(
                f"SHAP explanation unavailable: {shap_error}"
            )
        # -----------------------------
        # Normal Case
        # -----------------------------
        if disease == "Normal":

            st.markdown("---")

            st.success(
                "No autoimmune disease pattern detected."
            )

            st.info(
                "Patient appears healthy based on the available biomarkers."
            )

            st.stop()

        # -----------------------------
        # Disease Mapping
        # -----------------------------
        kb_mapping = {
            "Rheumatoid Arthritis":
                "Rheumatoid Arthritis (RA)",

            "Ankylosing Spondylitis":
                "Ankylosing Spondylitis (AS)",

            "Sjogren Syndrome":
                "Sjogren's Syndrome (SS)",

            "Psoriatic Arthritis":
                "Psoriatic Arthritis (PsA)",

            "Reactive Arthritis":
                "Reactive Arthritis (ReA)",

            "Systemic Lupus Erythematosus":
                "Systemic Lupus Erythematosus (SLE)"
        }

        kb_name = kb_mapping.get(disease, disease)

        disease_info = kb[
            kb["Disease"].astype(str).str.strip()
            == str(kb_name).strip()
        ]

        st.markdown("---")

        if not disease_info.empty:

            row = disease_info.iloc[0]

            st.subheader("Disease Description")
            st.write(row["Description"])

            st.subheader("Common Symptoms")
            st.write(row["Common Symptoms"])

            st.subheader("Important Biomarkers")
            st.write(row["Important Biomarkers"])

            st.subheader("Recommended Specialist")
            st.write(row["Recommended Specialist"])

            st.subheader("Recommended Tests")
            st.write(row["Recommended Tests"])

            st.subheader("Possible Complications")
            st.write(row["Possible Complications"])

        else:

            st.warning(
                f"Disease '{disease}' not found in knowledge base."
            )

    except Exception as e:
        st.error(str(e))