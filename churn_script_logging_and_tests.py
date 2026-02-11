'''
Unit Tests script for churn_library.py

Author: Belen Esteve
Date: Feb 2026
'''
import os
import pytest

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

import churn_library as cls
import constants


@pytest.fixture(scope="module")
def csv_path():
    '''
    Returns the path to the bank csv file
    '''
    return constants.BANK_DATA_CSV_PATH


@pytest.fixture(scope="module")
def bank_df(csv_path):
    '''
    Returns a dataframe out of the first 50 samples from
    the bank csv file
    '''
    return pd.read_csv(csv_path).head(50)


@pytest.fixture(scope="module")
def param_grid():
    '''
    Retruns a param grid dictionary for Gris Search Algorithm
    '''
    return {"n_estimators": [10]}


def test_import(csv_path):
    '''
    Test data import - this example is completed for you
    to assist with the other test functions
    '''
    try:
        df = cls.import_data(csv_path)
        print("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        print("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        print(
            "Testing import_data: The file doesn't appear to have rows and columns"
        )
        raise err


def test_eda(bank_df):
    '''
    Test perform eda function
    '''
    cls.perform_eda(bank_df)

    eda_files = [
        constants.CHURN_LABELS_FIGURE,
        constants.CUSTOMER_AGE_FIGURE,
        constants.MARITAL_STATUS_FIGURE,
        constants.TOTAL_TRANS_CT_FIGURE,
        constants.HEATMAP_FIGURE,
    ]

    try:
        for file in eda_files:
            path = os.path.join(constants.EDA_FOLDER_PATH, file)
            assert os.path.isfile(path), f"File {path} does not exist"
        print("Testing perform_eda: SUCCESS")
    except AssertionError as err:
        print(f"Testing perform_eda: FAILED - {err}")
        raise err


def test_encoder_helper(bank_df):
    '''
    Test encoder helper
    '''
    df_encoded = cls.encoder_helper(
        bank_df,
        constants.CATEGORY_COLUMNS_LIST,
        response=constants.OUTPUT_COLUMN_NAME
    )

    try:
        for col in constants.CATEGORY_COLUMNS_LIST:
            encoded_col = f"{col}_{constants.OUTPUT_COLUMN_NAME}"
            assert encoded_col in df_encoded.columns, (
                f"{encoded_col} is not a dataframe column"
            )
            assert df_encoded[encoded_col].isnull().sum() == 0, (
                f"There are null values in the encoded column {encoded_col}"
            )
        print("Testing encoder_helper: SUCCESS")
    except AssertionError as err:
        print(f"Testing encoder_helper: FAIL - {err}")
        raise err


def test_perform_feature_engineering(bank_df):
    '''
    Test perform_feature_engineering
    '''
    x_train, x_test, y_train, y_test = cls.perform_feature_engineering(
        bank_df,
        response=constants.OUTPUT_COLUMN_NAME
    )

    try:
        assert x_train.shape[0] > 0, "x_train shape[0] cannot be 0"
        assert x_test.shape[0] > 0, "x_test shape[0] cannot be 0"
        assert y_train.shape[0] > 0, "y_train shape[0] cannot be 0"
        assert y_test.shape[0] > 0, "y_test shape[0] cannot be 0"

        assert x_train.shape[1] == x_test.shape[1], (f"x_train.shape[1]={
            x_train.shape[1]} and x_test.shape[1]={
            x_test.shape[1]}, " + "but they should be equal")
        assert y_train.ndim == 1, (
            f"y_train.dim should be 1 but it is {y_train.ndim}"
        )
        assert y_test.ndim == 1, (
            f"y_test.dim should be 1 but it is {y_test.ndim}"
        )
        print("Testing perform_feature_engineering: SUCCESS")
    except AssertionError as err:
        print(f"Testing perform_feature_engineering: FAIL - {err}")
        raise err


def test_train_models(bank_df):
    '''
    Test train_models
    '''
    x_train, x_test, y_train, y_test = cls.perform_feature_engineering(
        bank_df,
        response=constants.OUTPUT_COLUMN_NAME
    )

    cls.train_models(x_train, x_test, y_train, y_test)

    # Check model artifact
    model_path = os.path.join(
        constants.MODELS_FOLDER_PATH,
        constants.RFC_MODEL_NAME + '_' + constants.MODEL_FILENAME
    )

    figures_list = [
        constants.RESULTS_FOLDER_PATH +
        constants.LR_MODEL_NAME +
        '_' +
        constants.RFC_MODEL_NAME +
        '_' +
        constants.ROC_CURVES_FIGURE,
        constants.RESULTS_FOLDER_PATH +
        constants.LR_MODEL_NAME +
        '_' +
        constants.RFC_MODEL_NAME +
        '_best_' +
        constants.ROC_CURVES_FIGURE]

    try:
        assert os.path.isfile(model_path), (
            f"Model path {model_path} does not exist."
        )
        assert os.path.isdir(constants.RESULTS_FOLDER_PATH), (
            f"Results folder {constants.RESULTS_FOLDER_PATH} does not exist"
        )
        assert len(os.listdir(constants.RESULTS_FOLDER_PATH)) > 0, (
            f"No elements found in {constants.RESULTS_FOLDER_PATH}"
        )
        for file in figures_list:
            assert os.path.exists(file), f"File {file} does not exist"
        print("Testing train_models: SUCCESS")
    except AssertionError as err:
        print(f"Testing train_models: FAILED - {err}")
        raise err


def test_train_model(bank_df):
    '''
    Test train_model
    '''
    x_train, x_test, y_train, y_test = cls.perform_feature_engineering(
        bank_df,
        response=constants.OUTPUT_COLUMN_NAME
    )
    lrc = LogisticRegression(solver='lbfgs', max_iter=30)
    model_name = constants.LR_MODEL_NAME
    lrc_model_data = [lrc, model_name]
    lrc, _ = cls.train_model(
        lrc_model_data, x_train, x_test, y_train, y_test)

    files_list = [
        constants.MODELS_FOLDER_PATH +
        model_name + '_' + constants.MODEL_FILENAME,
    ]
    try:
        for file in files_list:
            assert os.path.exists(file), f"File {file} does not exist"
        print("Testing test_model: SUCCESS")
    except AssertionError as err:
        print(f"Testing test_model: FAIL - {err}")
        raise err


def test_save_figure(tmp_path):
    '''
    Test save_figure
    '''
    test_path = tmp_path / "test_plot.png"

    # Create dummy plot
    plt.plot([1, 2, 3], [1, 4, 9])
    cls.save_figure(str(test_path))

    try:
        assert test_path.exists(), f"Figure image {
            str(test_path)} does not exist"
        print("Testing save_figure: SUCCESS")
    except AssertionError as err:
        print(f"Testing save_figure: FAILED - {err}")
        raise err


def test_get_best_estimator_logistic():
    '''
    Test get_best_estimator for LogisticRegression model
    '''
    model = LogisticRegression()
    best = cls.get_best_estimator(model)

    try:
        assert best is model, (
            "Model and best model should be the same for LogisticRegression model"
        )
        print("Testing best_estimator (LogisticRegression): SUCCESS")
    except AssertionError as err:
        print(f"Testing best_estimator (LogisticRegression): FAIL - {err}")
        raise err


def test_get_best_estimator_gridsearch(param_grid):
    '''
    Test get_best_estimator for GridSearch model
    '''
    gs = GridSearchCV(
        RandomForestClassifier(random_state=constants.RANDOM_SEED),
        param_grid=param_grid,
        cv=2
    )

    # fake-fit
    x_values = [[0], [1], [0], [1]]
    y_labels = [0, 1, 0, 1]
    gs.fit(x_values, y_labels)

    best = cls.get_best_estimator(gs)

    try:
        assert hasattr(best, "predict"), (
            "Predict should be an attribute of the model"
        )
        print("Testing get_best_estimator (GridSearch): SUCCESS")
    except AssertionError as err:
        print(f"Testing get_best_estimator (GridSearch): FAIL - {err}")
        raise err


def test_feature_importance_plot(bank_df, tmp_path, param_grid):
    '''
    Test feature_importance_plot
    '''
    x_train, x_test, y_train, _ = cls.perform_feature_engineering(
        bank_df,
        response=constants.OUTPUT_COLUMN_NAME
    )

    rfc = RandomForestClassifier(n_estimators=10,
                                 random_state=constants.RANDOM_SEED)
    model = GridSearchCV(estimator=rfc,
                         param_grid=param_grid,
                         cv=5,
                         n_jobs=-1)
    model.fit(x_train, y_train)

    output_path = tmp_path / "feature_importance.png"

    cls.feature_importance_plot(model, x_test, str(output_path))

    try:
        assert output_path.exists(), f"Figure {output_path} was not created"
        print("Testing feature_importance_plot: SUCCESS")
    except AssertionError as err:
        print(f"Testing feature_importance_plot: FAIL - {err}")
        raise err


def test_explain_features_fail(bank_df):
    '''
    Test explain_features and fails for no GridSearch model
    '''
    x_train, x_test, y_train, _ = cls.perform_feature_engineering(
        bank_df,
        response=constants.OUTPUT_COLUMN_NAME
    )

    model = RandomForestClassifier(
        n_estimators=5,
        max_depth=3,
        random_state=constants.RANDOM_SEED
    )
    model.fit(x_train, y_train)

    try:
        model_data = (model, "rf_light_test")
        cls.explain_features(model_data, x_test)

        raise Exception
    except ValueError:
        print(
            "Testing explain_features failure: SUCCESS - " +
            "It raises an error for models different to GridSearch"
        )
    except Exception as err:
        print(
            "Testing explain_features failure: FAIL - " +
            "It should raise an error for models different to GridSearch"
        )
        raise err


def test_explain_features_works(bank_df, param_grid):
    '''
    Test explain_features and works for GridSearch model
    '''
    x_train, x_test, y_train, _ = cls.perform_feature_engineering(
        bank_df,
        response=constants.OUTPUT_COLUMN_NAME
    )

    rfc = RandomForestClassifier(
        n_estimators=5,
        max_depth=3,
        random_state=constants.RANDOM_SEED
    )
    model = GridSearchCV(
        estimator=rfc,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1
    )
    model.fit(x_train, y_train)

    try:
        model_data = [model, constants.RFC_MODEL_NAME]
        cls.explain_features(model_data, x_test)
        print("Testing explain_features working: SUCCESS")
    except ValueError as err:
        print(
            "Testing explain_features working: FAIL - " +
            "For GridSearchCV model, explain_features should work")
        raise err


def test_classification_report_image(bank_df):
    '''
    Test classification_report_image
    '''
    x_train, x_test, y_train, y_test = cls.perform_feature_engineering(
        bank_df,
        response=constants.OUTPUT_COLUMN_NAME
    )
    model = LogisticRegression(solver='lbfgs', max_iter=30)
    model.fit(x_train, y_train)

    best_estimator = cls.get_best_estimator(model)

    y_train_preds = best_estimator.predict(x_train)
    y_test_preds = best_estimator.predict(x_test)

    model_name = constants.LR_MODEL_NAME
    cls.classification_report_image(y_train,
                                    y_test,
                                    y_train_preds,
                                    y_test_preds,
                                    model_name)
    files_list = [
        constants.RESULTS_FOLDER_PATH + model_name
        + "_" + constants.CLASSIFICATION_REPORT_FIGURE
    ]

    try:
        for file in files_list:
            assert os.path.exists(file), f"File {file} does not exist"
        print("Testing classification_report_image: SUCCESS")
    except AssertionError as err:
        print(f"Testing classification_report_image: FAIL - {err}")
        raise err
