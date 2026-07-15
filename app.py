import streamlit as st
import pandas as pd
import joblib


# -----------------------------------
# Page Configuration
# -----------------------------------

st.set_page_config(
    page_title="Restaurant Sales Prediction",
    page_icon="🍽️",
    layout="centered"
)


# -----------------------------------
# Load Model and Encoders
# -----------------------------------

model = joblib.load("model.pkl")

day_encoder = joblib.load("day_encoder.pkl")

category_encoder = joblib.load("category_encoder.pkl")


# -----------------------------------
# Title
# -----------------------------------

st.title("🍽️ Restaurant Sales Prediction")

st.write(
    "Predict restaurant sales based on day, category, quantity, pricing and customer rating."
)


# -----------------------------------
# User Inputs
# -----------------------------------

day = st.selectbox(
    "Select Day",
    [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]
)


category = st.selectbox(
    "Select Food Category",
    [
        "Pizza",
        "Burger",
        "Biryani",
        "Pasta"
    ]
)


quantity = st.number_input(
    "Quantity",
    min_value=1,
    max_value=100,
    value=2
)


price_per_item = st.number_input(
    "Price Per Item",
    min_value=1,
    value=250
)


discount = st.number_input(
    "Discount",
    min_value=0,
    max_value=100,
    value=10
)


customer_rating = st.slider(
    "Customer Rating",
    min_value=1,
    max_value=5,
    value=4
)


# -----------------------------------
# Prediction
# -----------------------------------

if st.button("Predict Sales"):

    # Encode categorical values

    day_encoded = day_encoder.transform([day])[0]

    category_encoded = category_encoder.transform([category])[0]


    # Create dataframe

    input_data = pd.DataFrame(
        {
            "Day": [day_encoded],
            "Category": [category_encoded],
            "Quantity": [quantity],
            "Price_Per_Item": [price_per_item],
            "Discount": [discount],
            "Customer_Rating": [customer_rating]
        }
    )


    # Prediction

    prediction = model.predict(input_data)


    st.success(
        f"Predicted Restaurant Sales: ₹ {prediction[0]:,.2f}"
    )


# -----------------------------------
# Footer
# -----------------------------------

st.markdown("---")

st.write(
    "Machine Learning Deployment Project using Random Forest Regression + Streamlit"
)