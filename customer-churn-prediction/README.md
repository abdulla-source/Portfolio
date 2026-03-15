# Customer Churn Prediction

This project builds a **machine learning model to predict customer churn** in a banking dataset. Customer churn occurs when a customer stops using a bank's services. Predicting churn helps financial institutions identify customers who are likely to leave and take preventive actions to retain them.

---

## Project Overview

The goal of this project is to develop an end-to-end machine learning pipeline that predicts whether a customer will exit the bank based on demographic, financial, and behavioral features.

The project includes:

- Exploratory Data Analysis (EDA)
- Feature Engineering
- Data Preprocessing
- Model Training
- Feature Importance Analysis
- Prediction on unseen test data

The model used in this project is **CatBoost**, a gradient boosting algorithm designed for tabular datasets and capable of handling categorical features efficiently.

---

## Dataset

The dataset contains information about bank customers, including:

- Credit score
- Geography
- Gender
- Age
- Account balance
- Number of bank products
- Credit card ownership
- Customer activity status
- Estimated salary

Target variable:

`Exited`

- `0` → customer stays with the bank  
- `1` → customer leaves the bank

---

## Feature Engineering

Several new features were created to capture customer behavior more effectively:

- **BalanceSalaryRatio** – relationship between account balance and estimated salary
- **EngagementScore** – customer interaction with bank services
- **AgeGroup** – customer age categorized into life stages

These engineered features help the model better understand patterns related to churn.

---

## Model

The model used for prediction is:

**CatBoostClassifier**

Reasons for choosing CatBoost:

- Handles categorical variables efficiently
- Performs well on structured datasets
- Robust to outliers and skewed data
- Requires minimal preprocessing

Evaluation metric:

**ROC-AUC**

The ROC-AUC score measures the model's ability to distinguish between customers who churn and those who remain.

---

## Results

Feature importance analysis shows that the most influential features include:

- Number of bank products
- Customer age
- Customer engagement
- Geography
- Account balance

These factors play a significant role in predicting whether a customer will leave the bank.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Seaborn
- Matplotlib
- Scikit-learn
- CatBoost

---

## Project Structure
