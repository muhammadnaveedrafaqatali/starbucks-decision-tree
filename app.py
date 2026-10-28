


import streamlit as st
import joblib
import pandas as pd

bundle = joblib.load("starbucks_calorie_model.pkl")
model = bundle["model"]
imputer = bundle["imputer"]
feature_cols = bundle["feature_cols"]

st.title("☕ Starbucks Calories Predictor")
st.write("Enter nutrition values to predict calories.")

# Example starting values for testing only
example_values = {
    "Total Fat (g)": 5.0,
    "Trans Fat (g)": 0.0,
    "Saturated Fat (g)": 3.0,
    "Sodium (mg)": 100.0,
    "Total Carbohydrates (g)": 20.0,
    "Cholesterol (mg)": 15.0,
    "Dietary Fibre (g)": 0.0,
    "Sugars (g)": 18.0,
    "Protein (g)": 2.0,
    "Caffeine (mg)": 75.0,
}

with st.form("prediction_form"):
    values = {}

    for col in feature_cols:
        default_value = example_values.get(col, 0.0)
        values[col] = st.number_input(
            col,
            min_value=0.0,
            value=float(default_value),
            step=0.1,
            key=col
        )

    submitted = st.form_submit_button("Predict Calories")

if submitted:
    input_df = pd.DataFrame([values])[feature_cols]
    input_clean = imputer.transform(input_df)
    prediction = model.predict(input_clean)[0]
    st.success(f"Predicted Calories: {prediction:.1f}")

