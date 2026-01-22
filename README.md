# Diabetes Prediction – MLOps Project

## Project Description
This project aims to build an end-to-end MLOps pipeline to predict whether a patient has diabetes using medical diagnostic measurements.  
The focus is on reproducibility, clean code structure, and MLOps best practices rather than model complexity.

## Machine Learning Task
- **Problem type:** Binary classification
- **Objective:** Predict diabetes outcome (0 = No diabetes, 1 = Diabetes)
- **Model (baseline):** Logistic Regression

## Dataset
- **Source:** https://github.com/plotly/datasets/blob/master/diabetes.csv
- **Features:** Medical attributes such as glucose level, BMI, age, etc.
- **Target:** `Outcome`

## Project Structure
DiabetesPrediction-mlops-project/
│
├── data/
│ └── diabetes.csv
├── src/
│ ├── data_loader.py
│ ├── preprocess.py
│ └── train.py
├── README.md
├── pyproject.toml
├── uv.lock
└── .gitignore