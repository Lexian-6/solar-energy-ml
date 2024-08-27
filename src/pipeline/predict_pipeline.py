import os
import pandas as pd

from src.exception import CustomizedException
from src.utils import load_object
from src.logger import logging

from dataclasses import dataclass


class PredictPipelineConfig:
    model_path = os.path.join("artifacts", "model.pkl")
    preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

class PredictPipeline:
    def __init__(self) -> None:
        self.predict_pipeline_config = PredictPipelineConfig()
    def predict(self, raw_features):
        try:
            model = load_object(self.predict_pipeline_config.model_path)
            preprocessor = load_object(self.predict_pipeline_config.preprocessor_path)
            logging.info("Load model and preprocessor.")
            processed_features = preprocessor.transform(raw_features)
            preds = model.predict(processed_features)
            logging.info("Make prediction.")
            return preds
        except Exception as e:
            logging.error(f"An error occurred: {CustomizedException(e)}")
            raise CustomizedException(e)
        
class CustomData:
    def __init__(self,
                 AmbTemp_C_Avg:float, 
                 WindSpeedAve_ms:float, 
                 WindDirAve_deg:float, 
                 RTD_C_Avg_Mean:float, 
                 Minute:float, ):
        self.AmbTemp_C_Avg = AmbTemp_C_Avg
        self.WindSpeedAve_ms = WindSpeedAve_ms
        self.WindDirAve_deg = WindDirAve_deg
        self.RTD_C_Avg_Mean = RTD_C_Avg_Mean
        self.Minute = Minute
    def get_data_as_data_frame(self):
        try:
            df=pd.DataFrame({
                "AmbTemp_C_Avg": [self.AmbTemp_C_Avg],
                "WindSpeedAve_ms": [self.WindSpeedAve_ms],
                "WindDirAve_deg": [self.WindDirAve_deg],
                "RTD_C_Avg_Mean": [self.RTD_C_Avg_Mean],
                "Minute": [self.Minute],
            })
            return df
        except Exception as e:
            logging.error(f"An error occurred: {CustomizedException(e)}")
            raise CustomizedException(e)