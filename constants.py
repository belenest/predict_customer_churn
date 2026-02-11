'''
Docstring for predict_customer_churn.constants

Author: Belen Esteve
Date: Feb 2026
'''
# Paths
HOME = './predict_customer_churn/'
BANK_DATA_CSV_PATH = f'{HOME}data/bank_data.csv'
RESULTS_FOLDER_PATH = f'{HOME}images/results/'
EDA_FOLDER_PATH = f'{HOME}images/eda/'
MODELS_FOLDER_PATH = f'{HOME}models/'
LOGS_PATH = f'{HOME}logs/churn_logs.log'

# Figure file names
CHURN_LABELS_FIGURE = 'churn_distribution.png'
CUSTOMER_AGE_FIGURE = 'customer_age_distribution.png'
MARITAL_STATUS_FIGURE = 'marital_status_distribution.png'
TOTAL_TRANS_CT_FIGURE = 'total_transaction_distribution.png'
HEATMAP_FIGURE = 'heatmap.png'
CLASSIFICATION_REPORT_FIGURE = 'classification_report.png'
ROC_CURVES_FIGURE = 'roc_curve_results.png'
LOGISTIC_REGRESSION_FIGURE = 'logistic_results.png'
RANDOM_FOREST_FIGURE = 'rf_results.png'
SHAP_FIGURE = 'shap_summary_plot.png'
FEATURE_IMPORTANCE_FIGURE = 'feature_importances.png'

# Model names
RFC_MODEL_NAME = 'random_forest'
LR_MODEL_NAME = 'logistic_regression'
MODEL_FILENAME = 'model.pkl'

# Other variables
RANDOM_SEED = 424
OUTPUT_COLUMN_NAME = 'Churn'

# Lists
CATEGORY_COLUMNS_LIST = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
]

QUANT_COLUMNS_LIST = [
    'Customer_Age',
    'Dependent_count',
    'Months_on_book',
    'Total_Relationship_Count',
    'Months_Inactive_12_mon',
    'Contacts_Count_12_mon',
    'Credit_Limit',
    'Total_Revolving_Bal',
    'Avg_Open_To_Buy',
    'Total_Amt_Chng_Q4_Q1',
    'Total_Trans_Amt',
    'Total_Trans_Ct',
    'Total_Ct_Chng_Q4_Q1',
    'Avg_Utilization_Ratio'
]
