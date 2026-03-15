# Bank Customer Churn Risk Dashboard

An interactive machine learning app that predicts whether a bank customer is at risk of churning. The project combines data analysis, feature engineering, model training, and deployment in a business-oriented dashboard.

## Overview

This project was built to solve a customer retention problem in banking. The model estimates churn probability from customer demographic, financial, and engagement-related information, then presents the result in a usable dashboard.

The app allows a user to:

- enter customer information
- generate churn probability
- view risk level
- see likely churn drivers
- get suggested retention actions

## Features

- End-to-end churn prediction workflow
- CatBoost model for tabular classification
- Feature engineering for better prediction quality
- Interactive Streamlit dashboard
- Business-friendly output and recommendations
- Example customer profiles for quick testing

## Machine Learning Workflow

The project includes:

- Problem definition
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Data Preprocessing
- Model Training with CatBoost
- Feature Importance Analysis
- Test Prediction
- Streamlit Deployment

## Engineered Features

The dashboard automatically creates additional features before prediction:

- **BalanceSalaryRatio**: balance relative to estimated salary
- **EngagementScore**: product usage × activity status
- **AgeGroup**: grouped age category

## Model

The main model used in this project is:

- **CatBoostClassifier**

Why CatBoost:

- strong performance on tabular data
- handles categorical variables well
- robust to skewed and non-linear relationships
- requires limited preprocessing

## Input Features

The model uses the following customer information:

- CreditScore
- Geography
- Gender
- Age
- Tenure
- Balance
- NumOfProducts
- HasCrCard
- IsActiveMember
- EstimatedSalary

## Output

The dashboard returns:

- churn probability
- risk level: Low / Medium / High
- top likely drivers
- suggested retention actions

## Project Structure

```text
customer-churn-prediction/
├── streamlit_churn_app.py
├── churn_model.pkl
├── requirements.txt
├── customer_churn.ipynb
└── README.md
