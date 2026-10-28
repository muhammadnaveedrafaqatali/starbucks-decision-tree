
import streamlit as st
import joblib
import pandas as pd

bundle = joblib.load("starbucks_calorie_model.pkl")
model = bundle["model"]
imputer = bundle["imputer"]
feature_cols = bundle["feature_cols"]

st.title("☕ Starbucks Calories Predictor")
st.write("Enter nutrition values to predict calories.")

with st.form("prediction_form"):
    values = {}
    for col in feature_cols:
        values[col] = st.number_input(
            col, min_value=0.0, value=0.0
        )

    submitted = st.form_submit_button("Predict Calories")

if submitted:
    input_df = pd.DataFrame([values])[feature_cols]
    input_clean = imputer.transform(input_df)
    prediction = model.predict(input_clean)[0]
    st.success(f"Predicted Calories: {prediction:.1f}")
