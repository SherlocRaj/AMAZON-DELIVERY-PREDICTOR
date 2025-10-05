import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="Amazon Delivery Time Predictor",
    page_icon="🚚",
    layout="wide"
)

# --- Model Loading ---
# Use st.cache_resource to load the model only once
@st.cache_resource
def load_model():
    """Loads the pre-trained model pipeline."""
    try:
        model = joblib.load('delivery_time_predictor.joblib')
        return model
    except FileNotFoundError:
        st.error("Model file not found. Make sure 'delivery_time_predictor.joblib' is in the same directory.")
        return None

model = load_model()

# --- User Interface ---
st.title("🚚 Amazon Delivery Time Predictor")
st.markdown("Enter the details of the delivery to get an estimated time of arrival.")

# Create columns for layout
col1, col2 = st.columns(2)

with col1:
    st.header("Order & Location Details")
    distance_km = st.number_input('Distance (km)', min_value=0.0, max_value=50.0, value=10.0, step=0.5)
    preparation_time_min = st.number_input('Preparation Time (minutes)', min_value=5.0, max_value=60.0, value=15.0, step=1.0)
    
    area = st.selectbox('Area', ['Urban', 'Metropolitian', 'Semi-Urban'])
    traffic = st.selectbox('Traffic', ['Low', 'Medium', 'High', 'Jam'])
    weather = st.selectbox('Weather Conditions', ['Sunny', 'Cloudy', 'Windy', 'Fog', 'Stormy', 'Sandstorms'])

with col2:
    st.header("Agent & Vehicle Details")
    agent_age = st.slider('Agent Age', 20, 50, 30)
    agent_rating = st.slider('Agent Rating', 1.0, 5.0, 4.5, 0.1)
    
    vehicle = st.selectbox('Vehicle Type', ['motorcycle', 'scooter', 'electric_scooter'])
    category = st.selectbox('Order Category', ['Snack', 'Drinks', 'Meal', 'Buffet'])
    is_weekend = st.checkbox('Is it a weekend?', value=False)


# --- Prediction Logic ---
if st.button('Predict Delivery Time', use_container_width=True):
    if model is not None:
        # Create a DataFrame from the user inputs
        # Column names must exactly match the ones used during training
        input_data = {
            'agent_age': [agent_age],
            'agent_rating': [agent_rating],
            'distance_km': [distance_km],
            'preparation_time_min': [preparation_time_min],
            'weather': [weather],
            'traffic': [traffic],
            'vehicle': [vehicle],
            'category': [category],
            'area': [area],
            'is_weekend': [1 if is_weekend else 0]
        }
        input_df = pd.DataFrame(input_data)
        
        # Make prediction
        prediction = model.predict(input_df)
        predicted_time = int(round(prediction[0]))
        
        # Display the result
        st.success(f"🎉 **Estimated Delivery Time: {predicted_time} minutes**")
    else:
        st.warning("Model is not loaded. Please check the file path.")