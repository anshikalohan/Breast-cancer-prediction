import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Breast Cancer Predictor", layout="centered")

# Load model and scaler 
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load("breast_cancer_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model_and_scaler()

default_values = {
    'radius_mean': 14.13, 'texture_mean': 19.29, 'perimeter_mean': 91.97,
    'area_mean': 654.89, 'smoothness_mean': 0.096, 'compactness_mean': 0.104,
    'concavity_mean': 0.089, 'concave points_mean': 0.048, 'symmetry_mean': 0.181,
    'fractal_dimension_mean': 0.063, 'radius_se': 0.405, 'texture_se': 1.217,
    'perimeter_se': 2.866, 'area_se': 40.34, 'smoothness_se': 0.007,
    'compactness_se': 0.025, 'concavity_se': 0.032, 'concave points_se': 0.012,
    'symmetry_se': 0.021, 'fractal_dimension_se': 0.004, 'radius_worst': 16.27,
    'texture_worst': 25.68, 'perimeter_worst': 107.26, 'area_worst': 880.58,
    'smoothness_worst': 0.132, 'compactness_worst': 0.254, 'concavity_worst': 0.272,
    'concave points_worst': 0.115, 'symmetry_worst': 0.290, 'fractal_dimension_worst': 0.084
}


feature_names = list(default_values.keys())

# Sidebar Inputs 
st.sidebar.header("Enter Tumor Features")
user_input = {}
for feature in feature_names:
    user_input[feature] = st.sidebar.number_input(
        feature, value=default_values[feature]
    )

# Title and Predict Button 
st.title("Breast Cancer Prediction Tool")

if st.button("Predict"):
    input_df = pd.DataFrame([user_input])
    scaled_input = scaler.transform(input_df)
    
    prediction = model.predict(scaled_input)[0]
    prediction_proba = model.predict_proba(scaled_input)[0]

    st.markdown("### Prediction Result")

    if prediction == 0:
        st.success("The tumor is **Benign** (Non-cancerous).")
    else:
        st.error("The tumor is **Malignant** (Cancerous).")

    st.markdown(f"**Confidence:** {prediction_proba[prediction]*100:.2f}%")
    st.markdown("> Always consult medical professionals for diagnosis.")

    # Visualizations 
    st.markdown("### Visual Analysis")

    # Pie chart of prediction probability
    fig_pie = px.pie(
        names=["Benign", "Malignant"],
        values=prediction_proba,
        title="Prediction Probability Distribution",
        color_discrete_sequence=["lightgreen", "lightcoral"]
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    # Coefficient-based bar chart
    if hasattr(model, "coef_"):
        coefs = model.coef_[0]
        top_indices = np.argsort(np.abs(coefs))[-5:][::-1]
        top_features = [feature_names[i] for i in top_indices]
        top_weights = coefs[top_indices]

        fig_bar = px.bar(
            x=top_features,
            y=top_weights,
            title="Top 5 Influential Features (Logistic Regression Coefficients)",
            labels={"x": "Feature", "y": "Weight"},
            color=top_weights,
            color_continuous_scale="RdBu"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

st.info("""
 **Model Info**  
**Model:** Logistic Regression  
**Features:** 30 extracted features from breast mass images  
**Prediction Classes:**  
- **Benign (0):** Non-cancerous  
- **Malignant (1):** Cancerous  

**Disclaimer:** This is a demo tool. Always consult medical professionals.
""")
