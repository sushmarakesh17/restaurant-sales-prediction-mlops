import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -----------------------------------
# Load Dataset
# -----------------------------------
df = pd.read_csv("restaurant_sales.csv")


# -----------------------------------
# Data Preprocessing
# -----------------------------------

# Convert Date column
df["Date"] = pd.to_datetime(df["Date"])

# Drop unnecessary columns
df = df.drop(["Order_ID", "Date"], axis=1)


# Encode categorical columns
label_encoder_day = LabelEncoder()
label_encoder_category = LabelEncoder()

df["Day"] = label_encoder_day.fit_transform(df["Day"])
df["Category"] = label_encoder_category.fit_transform(df["Category"])


# -----------------------------------
# Split Features and Target
# -----------------------------------

X = df.drop("Sales", axis=1)
y = df["Sales"]


# -----------------------------------
# Train Test Split
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -----------------------------------
# Train Model
# -----------------------------------

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


# -----------------------------------
# Model Evaluation
# -----------------------------------

y_pred = model.predict(X_test)

print("Model Performance")
print("-----------------")
print("MAE :", mean_absolute_error(y_test, y_pred))
print("MSE :", mean_squared_error(y_test, y_pred))
print("R2 Score :", r2_score(y_test, y_pred))


# -----------------------------------
# Save Model and Encoders
# -----------------------------------

joblib.dump(model, "model.pkl")

joblib.dump(label_encoder_day, "day_encoder.pkl")
joblib.dump(label_encoder_category, "category_encoder.pkl")


print("\nModel trained successfully!")
print("Files created:")
print("1. model.pkl")
print("2. day_encoder.pkl")
print("3. category_encoder.pkl")