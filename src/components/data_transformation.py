import os
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from src.logger import logging
from src.exception import CustomizedException
from src.utils import save_object

# Following imports are for testing only
from data_ingestion import DataIngestion


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    def get_preprocessor_object(self):
        try:
            num_columns = ['AmbTemp_C_Avg','WindSpeedAve_ms','WindDirAve_deg','RTD_C_Avg_Mean','Minute']
            
            num_pipeline = Pipeline(
                steps = [("imputer", SimpleImputer(strategy = "median")),
                         ("scaler", StandardScaler())
                ]
            )
            
            logging.info("Set up pipelines for all columns.")
            
            preprocessor = ColumnTransformer(
                [("num_pipeline", num_pipeline, num_columns)]
            )
            
            return preprocessor
        except Exception as e:
            logging.error(f"An error occurs {CustomizedException(e)}")
            raise CustomizedException(e)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data.")
            
            preprocessor = self.get_preprocessor_object()
            logging.info("Get preprocessor.")
            
            target_column_name = "PwrMtrP_kW_Avg"
            input_feature_train_df = train_df.drop(columns = [target_column_name])
            input_feature_test_df = test_df.drop(columns = [target_column_name])
            target_feature_train_df = train_df[target_column_name]
            target_feature_test_df = test_df[target_column_name]
            
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)
            
            train_arr = np.c_[
                input_feature_train_arr, 
                np.array(target_feature_train_df)
            ]
            
            test_arr = np.c_[
                input_feature_test_arr, 
                np.array(target_feature_test_df)
            ]
            logging.info("Preprocess train and test data")
            
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor
            )
            return (
                train_arr,
                test_arr, 
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            logging.error(f"An error occurs {CustomizedException(e)}")
            raise CustomizedException(e)
        
if __name__ == "__main__":
    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()
    
    dtobj = DataTransformation()
    dtobj.initiate_data_transformation(train_path, test_path)