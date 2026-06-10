# Customer Intelligence Suite

A deep learning-based application that predicts customer churn and estimates customer salary using Artificial Neural Networks (ANNs).

The project provides two predictive analytics modules:

1. Customer Churn Prediction
2. Estimated Salary Prediction

Both models are deployed through an interactive Streamlit interface.

---

## Features

### Customer Churn Prediction

Predicts whether a customer is likely to leave the bank based on:

* Demographics
* Account information
* Product usage
* Banking activity

### Salary Prediction

Estimates a customer's salary using banking and customer-related attributes.

### Interactive Dashboard

* User-friendly Streamlit interface
* Real-time predictions
* Instant probability scores
* No coding required

---

## Project Structure

```text
customer-intelligence-suite/
│
├── app.py
│
├── model.h5
├── regression_model.h5
│
├── label_encoder.pkl
├── one_hot.pkl
├── scaler.pkl
│
├── label_encode.pkl
├── one_hott.pkl
├── scalerr.pkl
│
├── requirements.txt
└── README.md
```

---

## Tech Stack

### Frontend

* Streamlit

### Machine Learning

* TensorFlow
* Keras

### Data Processing

* Pandas
* NumPy
* Scikit-Learn

---

## Customer Churn Prediction

The classification model predicts whether a customer is likely to churn.

### Input Features

```text
Credit Score
Geography
Gender
Age
Tenure
Balance
Number of Products
Credit Card Status
Active Membership Status
Estimated Salary
```

### Output

```text
Customer Likely To Churn

or

Customer Not Likely To Churn
```

---

## Salary Prediction

The regression model predicts estimated salary based on customer attributes.

### Input Features

```text
Credit Score
Geography
Gender
Age
Tenure
Balance
Number of Products
Credit Card Status
Active Membership Status
Exited Status
```

### Output

```text
Predicted Estimated Salary
```

---

## Data Preprocessing

The project uses:

### Label Encoding

Used for:

```text
Gender
```

### One-Hot Encoding

Used for:

```text
Geography
```

### Feature Scaling

Applied using StandardScaler before model inference.

---

## Model Architecture

### Churn Prediction

```text
Input Layer
      ↓
Hidden Layers
      ↓
Sigmoid Output Layer
      ↓
Binary Classification
```

### Salary Prediction

```text
Input Layer
      ↓
Hidden Layers
      ↓
Linear Output Layer
      ↓
Regression Prediction
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/customer-intelligence-suite.git

cd customer-intelligence-suite
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run Application

```bash
streamlit run app.py
```

---

## Example Use Cases

* Customer retention analysis
* Banking analytics
* Customer segmentation
* Churn prevention strategies
* Revenue forecasting
* Customer value assessment

---

## Future Improvements

* Batch prediction support
* Explainable AI integration
* Feature importance visualization
* Customer risk scoring
* Cloud deployment
* Real-time API integration

---

## Notes

* Churn prediction is a binary classification problem.
* Salary prediction is a regression problem.
* Models are trained using TensorFlow/Keras.
* Encoders and scalers are stored separately for consistent inference.
