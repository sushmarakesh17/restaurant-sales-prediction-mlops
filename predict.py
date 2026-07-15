import pandas as pd
import joblib


# -----------------------------------
# Load Model and Encoders
# -----------------------------------

model = joblib.load("model.pkl")

day_encoder = joblib.load("day_encoder.pkl")

category_encoder = joblib.load("category_encoder.pkl")


# -----------------------------------
# Prediction Function
# -----------------------------------

def predict_sales(
    day,
    category,
    quantity,
    price_per_item,
    discount,
    customer_rating
):

    # Encode categorical values
    day_encoded = day_encoder.transform([day])[0]
    category_encoded = category_encoder.transform([category])[0]


    # Create input dataframe
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


    # Predict sales
    prediction = model.predict(input_data)


    return prediction[0]


# -----------------------------------
# Test Prediction
# -----------------------------------

if __name__ == "__main__":

    predicted_sales = predict_sales(
        day="Friday",
        category="Pizza",
        quantity=3,
        price_per_item=250,
        discount=15,
        customer_rating=5
    )


    print("Predicted Restaurant Sales:", round(predicted_sales, 2))