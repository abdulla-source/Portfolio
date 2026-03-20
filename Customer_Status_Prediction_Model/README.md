# StatusBoost: Multi-Class Status Prediction with CatBoost

StatusBoost is a machine learning project for predicting multiple status categories from structured tabular data using **CatBoostClassifier**. The project focuses on building a clean and reliable classification pipeline with feature engineering, categorical feature handling, validation, and probability-based submission generation.

## Project Overview

The goal of this project is to classify records into multiple status classes based on medical and tabular input features. Since the dataset contains both numerical and categorical variables, CatBoost was selected as the primary model because it handles categorical data efficiently without requiring heavy preprocessing such as one-hot encoding.

This project was also used to explore an important machine learning challenge: **class imbalance**. In addition to the baseline model, a weighted version of the model was tested to improve minority-class detection and compare the tradeoff between global accuracy and balanced performance across classes.

## Key Features

- Multi-class classification with CatBoost
- Native handling of categorical features
- Feature engineering from raw columns
- Validation split for model evaluation
- Early stopping to reduce overfitting
- Probability-based predictions for submission format
- Feature importance analysis
- Class imbalance experiments with weighted learning

## Dataset

The project uses three files:

- `train.csv.zip` — training data
- `test.csv` — test data
- `sample_submission.csv` — submission format template

The target variable is:

- `Status`

The model predicts class probabilities for the required status categories.

## Workflow

The pipeline is organized into the following stages:

1. **Data Loading**
   - Read training, test, and sample submission files
   - Extract compressed training data

2. **Data Cleaning**
   - Standardize column names
   - Clean target labels
   - Handle missing categorical values

3. **Feature Engineering**
   - Convert `Age` into `AgeYears` for better interpretability
   - Prepare numerical and categorical features separately

4. **Train/Validation Split**
   - Split the training set into train and validation subsets
   - Use stratification to preserve class distribution

5. **Model Training**
   - Train a CatBoost multi-class classifier
   - Apply early stopping with validation data

6. **Evaluation**
   - Measure validation accuracy
   - Measure macro F1-score
   - Generate a classification report
   - Inspect feature importance

7. **Submission Generation**
   - Predict class probabilities on the test set
   - Format predictions to match the sample submission file

## Model Choice

CatBoost was chosen because it is especially effective for structured datasets with mixed numerical and categorical features. It reduces preprocessing complexity while still capturing non-linear relationships and feature interactions.

Main model settings include:

- `loss_function='MultiClass'`
- `eval_metric='MultiClass'`
- early stopping
- fixed random seed for reproducibility

## Results

### Baseline Model
The baseline model achieved strong overall validation performance:

- **Validation Accuracy:** 0.8497
- **Validation Macro F1-score:** 0.6167

Classification report summary:

- Class `C` was predicted very well
- Class `D` showed solid performance
- Class `CL` had very low recall because it was the minority class

This revealed that the dataset is imbalanced and that the model was biased toward the dominant classes.

### Weighted Model Experiment
To improve minority-class detection, class weights were introduced.

Weighted model results:

- **Validation Accuracy:** 0.7790
- **Validation Macro F1-score:** 0.6183

This experiment significantly improved recall for class `CL`, but reduced overall accuracy. The result demonstrates a key tradeoff in imbalanced classification:

- the baseline model performed better globally
- the weighted model performed better on the minority class

## Feature Importance

The most influential features in the baseline model included:

- `Bilirubin`
- `AgeYears`
- `N_Days`
- `Prothrombin`
- `Platelets`

These features contributed the most to the model’s predictions and suggest that the classifier learned meaningful patterns from the dataset.

## Project Structure

```bash
.
├── train.csv.zip
├── test.csv
├── sample_submission.csv
├── notebook.ipynb
├── submission.csv
└── README.md
