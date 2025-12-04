# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 23:19:51 2025

@author: micha
"""

import streamlit as st
import pandas as pd
import pickle

# ----------------------------
# Load the saved model and metadata
# ----------------------------
with open("model.pkl", "rb") as f:
    saved = pickle.load(f)

model = saved["model"]
feature_cols = saved["feature_cols"]

# ----------------------------
# Streamlit app UI
# ----------------------------
st.title("Reunion Attendance Prediction Prototype")

st.write(
    """
    This prototype uses our final Random Forest model to estimate 
    each alum’s likelihood of attending reunion.

    Upload a CSV of invitees with the same structure as the data we used 
    to train the model (i.e., the same cleaned feature columns), and the 
    app will return a ranked list of alumni by predicted probability.
    """
)

uploaded_file = st.file_uploader("Upload alumni CSV", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV
    df = pd.read_csv(uploaded_file)

    st.subheader("Preview of uploaded data")
    st.dataframe(df.head())

    # Ensure the uploaded file contains all expected feature columns
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        st.error(
            "The uploaded file is missing the following required columns: "
            + ", ".join(missing_cols)
        )
    else:
        # Subset columns to match training set and preserve order
        X_new = df[feature_cols].copy()

        # ----------------------------
        # CLEANING STEPS (MATCH TRAINING)
        # ----------------------------

        # 1. Fill missing values with numeric placeholder (-1)
        X_new = X_new.fillna(-1)

        # 2. Convert object (string) columns to numeric category codes
        for col in X_new.select_dtypes(include=["object"]).columns:
            X_new[col] = X_new[col].astype("category").cat.codes

        # ----------------------------
        # Run predictions
        # ----------------------------
        probs = model.predict_proba(X_new)[:, 1]

        # Attach predictions back to original data
        results = df.copy()
        results["Predicted_Probability"] = probs

        # Sort results from most to least likely to attend
        results_sorted = results.sort_values(
            by="Predicted_Probability", ascending=False
        )

        st.subheader("Ranked Alumni by Likelihood of Attending")
        st.write(
            "Alumni below are ranked from highest to lowest predicted probability of attending."
        )
        st.dataframe(results_sorted.head(50))  # show top 50 results

        # Download button for full ranked list
        csv_out = results_sorted.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download full ranked list as CSV",
            data=csv_out,
            file_name="ranked_alumni_predictions.csv",
            mime="text/csv",
        )
else:
    st.info("Upload a CSV file above to see predictions.")
