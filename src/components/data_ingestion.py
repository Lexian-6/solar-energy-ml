import os
from src.exception import CustomizedException
from src.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import select_features

from dataclasses import dataclass

# @dataclass: a decorator that automatically adds special methods to your class, such as __init__(), __repr__(), __eq__(), and others.
# Helps to reduce boilerplate code
@dataclass
class DataIngestionConfig:
    data_dir_path:str = "artifacts"
    train_data_path:str = os.path.join("artifacts", "train.csv")
    test_data_path:str = os.path.join("artifacts", "test.csv")
    raw_data_path:str = os.path.join("artifacts", "data.csv")

class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Start initiating data ingestion.")
        try:
            df = pd.read_csv(os.path.join("notebook", "data", "onemin-Ground-2017-01-01.csv"))
            logging.info("Read the dataset as dataframe.")
            
            df = select_features(df)
            
            os.makedirs(self.ingestion_config.data_dir_path, exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, header=True, index=False)
            logging.info("Raw dataset saved to local directory.")
            
            train_data,  test_data = train_test_split(df, test_size=0.2, random_state=42)
            logging.info("Train test split completed")
            train_data.to_csv(self.ingestion_config.train_data_path, header=True, index=False)
            test_data.to_csv(self.ingestion_config.test_data_path, header=True, index=False)
            
            logging.info("Data ingestion completed.")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.error(f"An error occurs: {CustomizedException(e)}")
            raise CustomizedException(e)
            
if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()
    