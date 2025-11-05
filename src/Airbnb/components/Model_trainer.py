import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge

from Airbnb.exception import customexception
from Airbnb.logger import logging
from Airbnb.utils.utils import evaluate_model, save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("Artifacts", "Model.pkl")
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    def initate_model_training(self, train_array, test_array):
        try:
            logging.info("Splitting dependent and independent variables from train and test data")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test,  y_test  = test_array[:, :-1],  test_array[:, -1]
            models = {
                "LinearRegression":      LinearRegression(),
                "Lasso":                 Lasso(),
                "Ridge":                 Ridge(),
                "ElasticNet":            ElasticNet(),
                "RandomForestRegressor": RandomForestRegressor(n_jobs=-1, random_state=42),
                "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42),
                "CatBoostRegressor":     CatBoostRegressor(verbose=False, allow_writing_files=False, random_state=42),
            }
            model_report = evaluate_model(X_train, y_train, X_test, y_test, models)
            for name, m in model_report.items():
                logging.info(f"{name:>26} | r2={m['r2']:.4f} | rmse={m['rmse']:.4f} | mae={m['mae']:.4f}")
            best_model_name  = max(model_report, key=lambda k: model_report[k]["r2"])
            best_model_score = model_report[best_model_name]["r2"]
            best_model       = models[best_model_name]
            logging.info(f"Best Model: {best_model_name}, R2={best_model_score:.4f}")
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)
            return best_model_name, best_model_score
        except Exception as e:
            logging.exception("Exception occurred at Model Training")
            raise customexception(e, sys)
