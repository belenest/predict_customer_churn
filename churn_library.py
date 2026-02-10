'''
Module for the churn library

Author: Belen Esteve
Date: Feb 2026
'''
# pylint: disable=too-many-locals, too-many-statements, too-many-arguments, import-error

# import libraries
import os
import logging
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import RocCurveDisplay, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

sns.set()

from . import constants


os.environ['QT_QPA_PLATFORM'] = 'offscreen'

logging.basicConfig(
    filename=constants.LOGS_PATH,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def import_data(pth):
    '''
    Returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            csv_df: pandas dataframe
    '''
    try:
        csv_df = pd.read_csv(pth)
        logging.info("Data imported successfully from: %s", pth)

        return csv_df
    except FileNotFoundError as err:
        logging.error("File not found: %s", err)
        return None


def perform_eda(csv_df):
    '''
    Perform eda on df and save figures to images folder

    input:
            csv_df: pandas dataframe

    output:
            None
    '''
    logging.info("Dataframe head:\n%s", csv_df.head())

    logging.info("Dataframe shape:\n%s", csv_df.shape)
    logging.info("Missing values:\n%s", csv_df.isnull().sum())
    logging.info("Dataframe description:\n%s", csv_df.describe())

    csv_df[constants.OUTPUT_COLUMN_NAME] = csv_df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    plt.figure(figsize=(20, 10))
    csv_df[constants.OUTPUT_COLUMN_NAME].hist()
    plt.tight_layout()
    plt.savefig(f'{constants.EDA_FOLDER_PATH}churn.png')
    plt.close()

    plt.figure(figsize=(20, 10))
    csv_df['Customer_Age'].hist()
    plt.tight_layout()
    plt.savefig(f'{constants.EDA_FOLDER_PATH}customer_age.png')
    plt.close()

    plt.figure(figsize=(20, 10))
    csv_df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.tight_layout()
    plt.savefig(f'{constants.EDA_FOLDER_PATH}marital_status.png')
    plt.close()

    plt.figure(figsize=(20, 10))
    # Show distributions of 'Total_Trans_Ct' and add a smooth curve obtained
    # using a kernel density estimate
    sns.histplot(csv_df['Total_Trans_Ct'], stat='density', kde=True)
    plt.tight_layout()
    plt.savefig(f'{constants.EDA_FOLDER_PATH}total_trans_ct.png')
    plt.close()

    plt.figure(figsize=(20, 10))
    sns.heatmap(csv_df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.tight_layout()
    plt.savefig(f'{constants.EDA_FOLDER_PATH}heatmap.png')
    plt.close()


def encoder_helper(csv_df, category_lst, response='Churn'):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            csv_df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for
             naming variables or index labels column]

    output:
            csv_df: pandas dataframe with new columns for
    '''
    for category in category_lst:
        try:
            csv_df[f'{category}_{response}'] = csv_df[category].map(
                csv_df.groupby(category).mean()[response])
        except KeyError:
            logging.error("Column %s or %s not found in dataframe", category, response)

    return csv_df


def perform_feature_engineering(csv_df, response='Churn'):
    '''
    input:
              csv_df: pandas dataframe
              response: string of response name [optional argument that could be used for
              naming variables or index labels column]

    output:
              x_train: x training data
              x_test: x testing data
              y_train: labels training data
              y_test: labels testing data
    '''
    labels = csv_df[constants.OUTPUT_COLUMN_NAME]
    features = pd.DataFrame()

    # Encode cat columns
    csv_df = encoder_helper(csv_df, constants.CATEGORY_COLUMNS_LIST, response)

    keep_cols = constants.QUANT_COLUMNS_LIST + \
        [f'{col}_{response}' for col in constants.CATEGORY_COLUMNS_LIST]

    features[keep_cols] = csv_df[keep_cols]

    # train test split
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.3, random_state=42)

    return x_train, x_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    plt.figure(figsize=(15, 10))

    plt.text(
        0.01,
        0.85,
        'Random Forest - Train\n' +
        classification_report(
            y_train,
            y_train_preds_rf),
        fontdict={
            'size': 10},
        family='monospace')

    plt.text(
        0.01,
        0.45,
        'Random Forest - Test\n' +
        classification_report(
            y_test,
            y_test_preds_rf),
        fontdict={
            'size': 10},
        family='monospace')

    plt.text(
        0.5,
        0.85,
        'Logistic Regression - Train\n' +
        classification_report(
            y_train,
            y_train_preds_lr),
        fontdict={
            'size': 10},
        family='monospace')

    plt.text(
        0.5,
        0.45,
        'Logistic Regression - Test\n' +
        classification_report(
            y_test,
            y_test_preds_lr),
        fontdict={
            'size': 10},
        family='monospace')

    plt.axis('off')
    plt.tight_layout()

    plt.savefig(f'{constants.RESULTS_FOLDER_PATH}classification_report.png')
    plt.close()


def feature_importance_plot(model, x_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            x_data: pandas dataframe of x values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)
    plt.tight_layout()
    plt.savefig(output_pth)
    plt.close()


def train_models(x_train, x_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              x_train: x training data
              x_test: x testing data
              y_train: labels training data
              y_test: labels testing data
    output:
              None
    '''
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_train, y_train)

    lrc.fit(x_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)

    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

    plt.figure(figsize=(15, 8))
    lrc_plot = RocCurveDisplay.from_estimator(lrc, x_test, y_test)
    plt.tight_layout()
    plt.savefig(f'{constants.RESULTS_FOLDER_PATH}logistic_regression_roc_curve.png')
    plt.close()

    # plots
    plt.figure(figsize=(15, 8))
    axis = plt.gca()
    RocCurveDisplay.from_estimator(
        cv_rfc.best_estimator_,
        x_test,
        y_test,
        ax=axis,
        alpha=0.8)
    lrc_plot.plot(ax=axis, alpha=0.8)
    plt.tight_layout()
    plt.savefig(f'{constants.RESULTS_FOLDER_PATH}random_forest_roc_curve.png')
    plt.close()

    # save best model
    joblib.dump(cv_rfc.best_estimator_, f'{constants.MODELS_FOLDER_PATH}rfc_model.pkl')
    joblib.dump(lrc, f'{constants.MODELS_FOLDER_PATH}logistic_model.pkl')

    rfc_model = joblib.load(f'{constants.MODELS_FOLDER_PATH}rfc_model.pkl')
    lr_model = joblib.load(f'{constants.MODELS_FOLDER_PATH}logistic_model.pkl')

    lrc_plot = RocCurveDisplay.from_estimator(lr_model, x_test, y_test)

    plt.figure(figsize=(15, 8))
    axis = plt.gca()
    RocCurveDisplay.from_estimator(rfc_model, x_test, y_test, ax=axis, alpha=0.8)
    lrc_plot.plot(ax=axis, alpha=0.8)
    plt.tight_layout()
    plt.savefig(f'{constants.RESULTS_FOLDER_PATH}roc_curve.png')
    plt.close()

    explainer = shap.TreeExplainer(cv_rfc.best_estimator_)
    shap_values = explainer.shap_values(x_test)
    shap.summary_plot(shap_values, x_test, plot_type="bar")
    plt.savefig(f'{constants.RESULTS_FOLDER_PATH}shap_summary_plot.png')
    plt.close()

    feature_importance_plot(cv_rfc, x_test,
                            f'{constants.RESULTS_FOLDER_PATH}feature_importance.png')

    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')

    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{constants.RESULTS_FOLDER_PATH}classification_report.png')
    plt.close()


if __name__ == "__main__":
    bank_df = import_data(constants.BANK_DATA_CSV_PATH)

    perform_eda(bank_df)

    data_train, data_test, labels_train, labels_test = perform_feature_engineering(
        bank_df, response=constants.OUTPUT_COLUMN_NAME)
    train_models(data_train, data_test, labels_train, labels_test)
