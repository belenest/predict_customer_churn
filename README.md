# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This repo performs data analysis, trains a model and predicts the likelihood of the clients to be churn. 
This project builds a machine learning pipeline that:
- Performs exploratory data analysis (EDA)
- Conducts feature engineering
- Trains classification models
- Evaluates model performance
- Generates model interpretability outputs
- Logs execution steps
- Includes unit testing for production readiness

The goal is to predict whether a customer will churn based on historical banking data.

## Files and data description

### Project structure
```
predict_customer_churn/
│
├── data/
│   └── bank_data.csv
│
├── images/
│   ├── eda/
│   │   ├── churn_labels.png
│   │   ├── customer_age.png
│   │   ├── heatmap.png
│   │   ├── marital_status.png
│   │   └── total_trans_ct.png
│   │
│   └── results/
│       ├── logistic_regression_random_forest_best_roc_curve.png
│       ├── logistic_regression_random_forest_roc_curve.png
│       ├── random_forest_classification_report.png
│       ├── random_forest_feature_importance.png
│       └── random_forest_shap_summary_plot.png
│
├── logs/
│
├── models/
│
├── churn_library.py
├── churn_script_logging_and_tests.py
├── constants.py
├── requirements.in
├── requirements.txt
└── LICENSE
```

### Dataset
The dataset used in this project contains customer demographic information, account activity, and transaction behavior used to predict churn. It is accessible at:

```
data/bank_data.csv
```

## Running Files

### Installation
Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```
Install dependencies:

```bash
python -m pip install -r requirements.txt
```
### Execution

#### Main Pipeline
This performs:
- Data loading
- EDA
- Feature engineering
- Model training
- Evaluation
- Saving outputs
- Logging

```bash
python predict_customer_churn/churn_library.py
```

#### Unit Tests
There is a test per function in churn_library, and they can be executed using:

```bash
pytest -s predict_customer_churn/churn_script_logging_and_tests.py > predict_customer_churn/logs/test_logs.log
```

#### Logging
Logs are stored in /logs:

```
churn_logs.log # Outcome of main script
test_logs.log # Outcome of pytest
```