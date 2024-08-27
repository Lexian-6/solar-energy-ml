import os

from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomizedException

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.utils import evaluate_models, save_object

# Following imports are for testing only
from data_ingestion import DataIngestion
from data_transformation import DataTransformation

@dataclass
class ModelTrainerConfig:
    best_model_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            X_train = train_arr[:, :-1]
            X_test = test_arr[:, :-1]
            y_train = train_arr[:, -1]
            y_test = test_arr[:, -1]
            logging.info("Split training and test input data")
            
            models = {
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            
            params={
                "Decision Tree": {
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter':['best','random'],
                    'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    
                    'max_features':['sqrt','log2',None],
                    # 'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    # 'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    # 'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    # 'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    # 'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    # 'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    # 'n_estimators': [8,16,32,64,128,256]
                }
            }
            
            report, finetuned_models = evaluate_models(X_train, y_train, X_test, y_test, models, params)
            logging.info("Get model reports.")
            
            best_model_name = max(report)
            best_model_value = max(report.values())
            best_model = finetuned_models[best_model_name]
            
            if best_model_value < 0.6:
                error_message = f"Low accuracy for all models. Best model '{best_model_name}' has validation R2 score of only {best_model_value:.4f}"
                logging.warning(error_message)
                raise CustomizedException(error_message)
            logging.info(f"Best model being '{best_model_name}', with a validation R2 score of {best_model_value:.4f}")
            save_object(
                file_path=self.model_trainer_config.best_model_path,
                obj=best_model
            )
            
            y_pred = best_model.predict(X_test)
            r2_square = r2_score(y_true=y_test, y_pred=y_pred)
            
            return r2_square
        except Exception as e:
            logging.error(f"An error occurred: {CustomizedException(e)}")
            raise CustomizedException(e)
        
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))