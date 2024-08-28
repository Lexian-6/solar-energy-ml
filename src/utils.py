import os

import dill
import pandas as pd

from src.exception import CustomizedException
from src.logger import logging

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        logging.error(f"An error occurred: {CustomizedException(e)}")
        raise CustomizedException(e)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        logging.error(f"An error occurred: {CustomizedException(e)}")
        raise CustomizedException(e)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        finetuned_models = {}
        logging.info(f"Start perfoming exhaustive search over parameters to find the best model")
        for i in range(len(models)):
            model = list(models.values())[i]
            model_name = list(models.keys())[i]
            param = params[model_name]
            logging.info(f"Start finetuning {model_name}")
            
            # Grid Search perform exhaustive search over specified parameter values for a model (an estimator).
            # cv=5 suggests that we are using kfold cross validation with k=5
            gs = GridSearchCV(model, param, cv=5)
            gs.fit(X_train, y_train)
            
            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)
            
            y_test_pred = model.predict(X_test)
            model_score = r2_score(y_true=y_test, y_pred=y_test_pred)
            
            report[model_name] = model_score
            finetuned_models[model_name] = model
            logging.info(f"Finish finetuning {model_name}")
        return report, finetuned_models
        
    except Exception as e:
        logging.error(f"An error occurred: {CustomizedException(e)}")
        raise CustomizedException(e)
    
def select_features(df):
    # List of features we want to keep
    features = [
        'TIMESTAMP',
        'AmbTemp_C_Avg',
        'WindSpeedAve_ms',
        'WindDirAve_deg',
        'RTD_C_Avg_1',
        'RTD_C_Avg_2',
        'RTD_C_Avg_3',
        'RTD_C_Avg_4',
        'RTD_C_Avg_5',
        'RTD_C_Avg_6',
        'RTD_C_Avg_7',
        'RTD_C_Avg_8',
        'RTD_C_Avg_9',
        'RTD_C_Avg_10'
    ]

    # Target variable
    target = 'PwrMtrP_kW_Avg'

    # Create a new dataframe with only the features and target variable we want
    df_selected = df[features + [target]]

    # Using mean
    df_selected['WindSpeedAve_ms'].fillna(method = 'bfill', inplace=True)

    # List of RTD columns
    rtd_columns = ['RTD_C_Avg_1', 'RTD_C_Avg_2', 'RTD_C_Avg_3', 'RTD_C_Avg_4', 'RTD_C_Avg_5', 
                'RTD_C_Avg_6', 'RTD_C_Avg_7', 'RTD_C_Avg_8', 'RTD_C_Avg_9', 'RTD_C_Avg_10']

    # Create a new column with the average of all RTD readings
    df_selected['RTD_C_Avg_Mean'] = df_selected[rtd_columns].mean(axis=1)

    # If you want to drop the original RTD columns:
    df_selected = df_selected.drop(columns=rtd_columns)

    # Assuming your dataframe is named df_selected and the timestamp column is 'TIMESTAMP'
    df_selected['TIMESTAMP'] = pd.to_datetime(df_selected['TIMESTAMP'])

    # Extract the minute component
    df_selected['Minute'] = df_selected['TIMESTAMP'].dt.hour * 60 + df_selected['TIMESTAMP'].dt.minute

    df_selected = df_selected.drop("TIMESTAMP", axis=1)
    
    return df_selected
    