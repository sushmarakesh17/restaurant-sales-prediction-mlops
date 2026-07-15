# 🍽️ Restaurant Sales Prediction - MLOps

An end-to-end Machine Learning Operations (MLOps) project that predicts restaurant sales using a Random Forest Regression model. The application is deployed using Streamlit, allowing users to estimate sales based on restaurant-related input features.

## 🚀 Live Demo

🔗 https://restaurant-sales-prediction-mlops-5fprjqqyykzqjwtb28u737.streamlit.app/

---

## 📌 Project Overview

Restaurant businesses generate large volumes of sales data every day. Predicting future sales helps in inventory planning, staffing, pricing, and overall business decision-making.

This project demonstrates the complete machine learning workflow—from data preprocessing and model training to deployment as an interactive web application.

---

## ✨ Features

* Predict restaurant sales instantly
* Interactive Streamlit web interface
* Data preprocessing and encoding
* Random Forest Regression model
* Model serialization using Joblib
* User-friendly prediction interface
* End-to-end ML deployment

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Streamlit

---

## 📂 Project Structure

```text
restaurant-sales-prediction-mlops/
│
├── restaurant_sales.csv
├── train.py
├── predict.py
├── app.py
├── model.pkl
├── day_encoder.pkl
├── category_encoder.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📊 Input Features

* Day
* Food Category
* Quantity
* Price Per Item
* Discount
* Customer Rating

### 🎯 Target Variable

* Sales

---

## 🤖 Machine Learning Model

* Algorithm: Random Forest Regressor
* Train-Test Split
* Label Encoding for categorical variables
* Model saved using Joblib

---

## ▶️ How to Run the Project

### Clone the Repository

```bash
git clone https://github.com/sushmarakesh17/restaurant-sales-prediction-mlops.git
```

### Navigate to the Project Folder

```bash
cd restaurant-sales-prediction-mlops
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train the Model

```bash
python train.py
```

### Run the Streamlit App

```bash
streamlit run app.py
```

---

## 🌐 Live Application

https://restaurant-sales-prediction-mlops-5fprjqqyykzqjwtb28u737.streamlit.app/

---

## 📸 Application Preview

*Add a screenshot of your Streamlit application here.*

---

## 📈 Future Enhancements

* Docker Deployment
* MLflow Integration
* CI/CD Pipeline
* Cloud Deployment (AWS/Azure/GCP)
* Model Monitoring
* Automated Retraining

---

## 👩‍💻 Author

**Sushma Rakesh**

GitHub: https://github.com/sushmarakesh17

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.
