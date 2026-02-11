'''
Module for the churn library

Author: Belen Esteve
Date: Feb 2026
'''
# pylint: disable=too-many-locals, too-many-statements,
# too-many-arguments, import-error

# import libraries
import os
import logging
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

from sklearn.metrics import RocCurveDisplay, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import constants

sns.set()
matplotlib.use("Agg")

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

    logging.info("Attrition_flag")
    logging.info(csv_df['Attrition_Flag'])

    csv_df[constants.OUTPUT_COLUMN_NAME] = csv_df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    logging.info("Attrition_flag")
    logging.info(csv_df['Attrition_Flag'])

    # Ensure the directory exists or create it
    if not os.path.exists(constants.EDA_FOLDER_PATH):
        os.makedirs(constants.EDA_FOLDER_PATH)

    plt.figure(figsize=(20, 10))
    csv_df[constants.OUTPUT_COLUMN_NAME].hist()
    save_figure(constants.EDA_FOLDER_PATH + constants.CHURN_LABELS_FIGURE)

    plt.figure(figsize=(20, 10))
    csv_df['Customer_Age'].hist()
    save_figure(constants.EDA_FOLDER_PATH + constants.CUSTOMER_AGE_FIGURE)

    plt.figure(figsize=(20, 10))
    csv_df.Marital_Status.value_counts('normalize').plot(kind='bar')
    save_figure(constants.EDA_FOLDER_PATH + constants.MARITAL_STATUS_FIGURE)

    plt.figure(figsize=(20, 10))
    # Show distributions of 'Total_Trans_Ct' and add a smooth curve obtained
    # using a kernel density estimate
    sns.histplot(csv_df['Total_Trans_Ct'], stat='density', kde=True)
    save_figure(constants.EDA_FOLDER_PATH + constants.TOTAL_TRANS_CT_FIGURE)

    plt.figure(figsize=(20, 10))
    numeric_df = csv_df.select_dtypes(include='number')
    sns.heatmap(numeric_df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    save_figure(constants.EDA_FOLDER_PATH + constants.HEATMAP_FIGURE)


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
                csv_df.groupby(category)[response].mean())
        except KeyError:
            logging.error(
                "Column %s or %s not found in dataframe",
                category,
                response)

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
        features, labels, test_size=0.3, random_state=constants.RANDOM_SEED)

    return x_train, x_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds,
                                y_test_preds,
                                model_name):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds: training predictions from a model
            y_test_preds: test predictions from a model
            model_name: str with the model name

    output:
             None
    '''
    plt.figure(figsize=(5, 5))
    plt.text(0.01, 1.25, str(f'{model_name} Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str(f'{model_name} Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')

    save_figure(constants.RESULTS_FOLDER_PATH + model_name +
                "_" + constants.CLASSIFICATION_REPORT_FIGURE)


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
    best_estimator = get_best_estimator(model)
    # Calculate feature importances
    importances = best_estimator.feature_importances_
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
    save_figure(output_pth)


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
    # Create results and model folders if they do not not exist
    if not os.path.exists(constants.RESULTS_FOLDER_PATH):
        os.makedirs(constants.RESULTS_FOLDER_PATH)

    if not os.path.exists(constants.MODELS_FOLDER_PATH):
        os.makedirs(constants.MODELS_FOLDER_PATH)

    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
    lrc_model_data = [lrc, constants.LR_MODEL_NAME]
    lrc, lrc_best_model = train_model(
        lrc_model_data, x_train, x_test, y_train, y_test)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    # grid search
    rfc = RandomForestClassifier(random_state=constants.RANDOM_SEED)
    cv_rfc = GridSearchCV(estimator=rfc,
                          param_grid=param_grid,
                          cv=5,
                          n_jobs=-1)
    cv_rfc_model_data = [cv_rfc, constants.RFC_MODEL_NAME]
    cv_rfc, rfc_best_model = train_model(
        cv_rfc_model_data, x_train, x_test, y_train, y_test)

    # Plot with models
    lrc_plot = RocCurveDisplay.from_estimator(lrc, x_test, y_test)
    plt.figure(figsize=(15, 8))
    axis = plt.gca()
    RocCurveDisplay.from_estimator(
        cv_rfc.best_estimator_,
        x_test,
        y_test,
        ax=axis,
        alpha=0.8)
    lrc_plot.plot(ax=axis, alpha=0.8)
    save_figure(
        constants.RESULTS_FOLDER_PATH +
        constants.LR_MODEL_NAME +
        '_' +
        constants.RFC_MODEL_NAME +
        '_' +
        constants.ROC_CURVES_FIGURE)

    # Plot with best models
    lrc_plot = RocCurveDisplay.from_estimator(lrc_best_model, x_test, y_test)

    plt.figure(figsize=(15, 8))
    axis = plt.gca()
    RocCurveDisplay.from_estimator(
        rfc_best_model, x_test, y_test, ax=axis, alpha=0.8)
    lrc_plot.plot(ax=axis, alpha=0.8)
    save_figure(
        constants.RESULTS_FOLDER_PATH +
        constants.LR_MODEL_NAME +
        '_' +
        constants.RFC_MODEL_NAME +
        '_best_' +
        constants.ROC_CURVES_FIGURE)


def train_model(model_data, x_train, x_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
            model_data: duple of model to be trained and its name
            x_train: x training data
            x_test: x testing data
            y_train: labels training data
            y_test: labels testing data
    output:
            model: trained_model
            best_model: selected best model
    '''
    model, model_name = model_data

    # Train model
    model.fit(x_train, y_train)

    best_estimator = get_best_estimator(model)

    y_train_preds = best_estimator.predict(x_train)
    y_test_preds = best_estimator.predict(x_test)

    # scores
    logging.info('%s results', model_name)
    logging.info('test results')
    logging.info(classification_report(y_test, y_test_preds))
    logging.info('train results')
    logging.info(classification_report(y_train, y_train_preds))

    # save best model
    model_file_path = constants.MODELS_FOLDER_PATH + \
        model_name + '_' + constants.MODEL_FILENAME

    joblib.dump(best_estimator, model_file_path)
    best_model = joblib.load(model_file_path)

    # Explain the features only for Grid Search (not allowed for Logistic
    # Regression)
    if isinstance(model, GridSearchCV):
        explain_features(model_data, x_test)

        classification_report_image(y_train,
                                    y_test,
                                    y_train_preds,
                                    y_test_preds,
                                    model_name=model_name)

    return model, best_model


def explain_features(model_data, x_test):
    '''
    Obtains the SHAP values of a model and saves its feature importance

    input:
            model_data: duple of model to be trained and its name
            x_test: x testing data
    output:
            None
    '''
    model, model_name = model_data

    if not isinstance(model, GridSearchCV):
        raise ValueError(
            f"Undifined model type {
                type(model)} for feature explanation")

    best_estimator = get_best_estimator(model)
    explainer = shap.TreeExplainer(best_estimator)
    shap_values = explainer.shap_values(x_test)
    shap.summary_plot(
        shap_values,
        x_test,
        plot_type="bar",
        max_display=10,
        show=False)
    save_figure(
        constants.RESULTS_FOLDER_PATH +
        model_name +
        '_' +
        constants.SHAP_FIGURE)

    feature_importance_plot(model, x_test,
                            constants.RESULTS_FOLDER_PATH + model_name + '_'
                            + constants.FEATURE_IMPORTANCE_FIGURE)


def save_figure(output_path):
    '''
    Helper function to save a figure in the indicated path

    input:
            output_path: str to the path where the image will be stored

    output:
            None
    '''
    fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def get_best_estimator(model):
    '''
    Returns the model best estimator depending on its class

    input:
            model: model to get the best estimator from
    output:
            best_estimator: best estimator of the model
    '''
    if isinstance(model, LogisticRegression):
        best_estimator = model
    elif isinstance(model, GridSearchCV):
        best_estimator = model.best_estimator_
    else:
        raise ValueError(f"Undifined model type {type(model)}")

    return best_estimator


if __name__ == "__main__":
    bank_df = import_data(constants.BANK_DATA_CSV_PATH)

    perform_eda(bank_df)

    data_train, data_test, labels_train, labels_test = perform_feature_engineering(
        bank_df, response=constants.OUTPUT_COLUMN_NAME)
    train_models(data_train, data_test, labels_train, labels_test)
