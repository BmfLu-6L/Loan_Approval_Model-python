import os
import pickle
import streamlit as st
import pandas as pd
import numpy as np

# Load dataset
artifacts_path_pipe = os.path.join(os.path.dirname(__file__), 'pipe.pkl')
artifacts_path_df = os.path.join(os.path.dirname(__file__), 'df.pkl')

try:
    with open(artifacts_path_pipe, 'rb') as file:
        pipe = pickle.load(file)

    with open(artifacts_path_df, 'rb') as file:
        df_info = pickle.load(file)

    loaded_model = pipe['model']
    X_train_columns = pipe['X_train_columns']
    scaler = df_info['scaler']
    imputer_num = df_info['imputer_num']
    imputer_cat = df_info['imputer_cat']
    categorical_cols = df_info['categorical_cols']
    numerical_cols = df_info['numerical_cols']
    unique_vals_dict = df_info['unique_vals']

except FileNotFoundError:
    st.error("pkl files not found. Please ensure they are in the same directory as app.py.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred loading pkl files: {e}")
    st.stop()

# App title
st.title("Loan Default Prediction App")

# Data input form
input_data = {}

col1, col2 = st.columns(2)

with col1:
    for col in numerical_cols[:len(numerical_cols) // 2]:
        if col in ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'cb_person_cred_hist_length']:
            input_data[col] = st.number_input(col.replace('_', ' ').title(), value=0, step=1)
        else:
            input_data[col] = st.number_input(col.replace('_', ' ').title(), value=0.0)

    for i, col in enumerate(categorical_cols):
        unique_vals = unique_vals_dict.get(col)
        if unique_vals:
            input_data[col] = st.selectbox(col.replace('_', ' ').title(), unique_vals, key=f"selectbox_{col}_{i}")
        else:
            st.write(f"Warning: Unique values not found for {col}.")
            input_data[col] = None

with col2:
    for col in numerical_cols[len(numerical_cols) // 2:]:
        if col in ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'cb_person_cred_hist_length']:
            input_data[col] = st.number_input(col.replace('_', ' ').title(), value=0, step=1)
        else:
            input_data[col] = st.number_input(col.replace('_', ' ').title(), value=0.0)




# Button
if st.button("Predict"):

    # Prediction
    input_df = pd.DataFrame([input_data])


    for col in numerical_cols:
        if col in ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'cb_person_cred_hist_length']:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce').astype('Int64')
        else:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

    input_df[numerical_cols] = imputer_num.transform(input_df[numerical_cols])
    input_df[categorical_cols] = imputer_cat.transform(input_df[categorical_cols])

    input_df = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)

    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    for col in X_train_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[X_train_columns]

    try:
        prediction_proba = loaded_model.predict_proba(input_df)[:, 1][0] if hasattr(loaded_model, "predict_proba") else \
            loaded_model.predict(input_df)[0]
        prediction_label = (prediction_proba >= 0.5).astype(int) if hasattr(loaded_model,
                                                                            "predict_proba") else loaded_model.predict(
            input_df)

        st.write(f"Prediction Probability (Default): {prediction_proba:.4f}")
        st.write(f"Prediction Label (Default/Not Default): {prediction_label}")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")