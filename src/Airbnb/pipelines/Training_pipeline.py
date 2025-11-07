import os

import pandas as pd

from Airbnb.components.Data_ingestion import DataIngestion
from Airbnb.components.Data_transformation import DataTransformation
from Airbnb.components.HyperparameterTuner import run_optuna_search
from Airbnb.components.Model_trainer import ModelTrainer
from Airbnb.components.PerCityTrainer import train_per_city

if __name__ == "__main__":
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()
    train_df = pd.read_csv(train_data_path)
    test_df  = pd.read_csv(test_data_path)
    data_transformation = DataTransformation()
    train_arr, test_arr = data_transformation.initialize_data_transformation(train_data_path, test_data_path)
    if os.environ.get("AIRBNB_RUN_OPTUNA") == "1":
        X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
        X_test,  y_test  = test_arr[:, :-1],  test_arr[:, -1]
        best_name, _, best_params = run_optuna_search(
            X_train, y_train, X_test, y_test,
            n_trials=int(os.environ.get("OPTUNA_TRIALS", "25")),
        )
        if best_name is not None:
            print(f"Optuna best model: {best_name} params={best_params}")
        else:
            print("Optuna search unavailable; falling back to default trainer.")
            model_trainer = ModelTrainer()
            best_name, best_score = model_trainer.initate_model_training(train_arr, test_arr)
            print(f"Default trainer best: {best_name} r2={best_score:.4f}")
    else:
        model_trainer = ModelTrainer()
        best_name, best_score = model_trainer.initate_model_training(train_arr, test_arr)
        print(f"Training complete. Best model: {best_name} (R2={best_score:.4f})")
    if os.environ.get("AIRBNB_DISABLE_PER_CITY") != "1":
        try:
            from Airbnb.components.Data_transformation import (
                add_seasonality_features,
                parse_host_response_rate,
                vectorize_amenity_count,
            )
            train_clean = train_df.copy()
            test_clean  = test_df.copy()
            train_clean['host_response_rate'] = parse_host_response_rate(train_clean['host_response_rate'])
            test_clean ['host_response_rate'] = parse_host_response_rate(test_clean ['host_response_rate'])
            train_clean['amenities'] = vectorize_amenity_count(train_clean['amenities'])
            test_clean ['amenities'] = vectorize_amenity_count(test_clean ['amenities'])
            train_clean = add_seasonality_features(train_clean)
            test_clean  = add_seasonality_features(test_clean)
            metrics = train_per_city(train_clean, test_clean)
            print("Per-city metrics:")
            for city, m in metrics.items():
                line = f"  {city}: {m['status']}"
                if 'r2' in m:
                    line += f"  r2={m['r2']:.4f}"
                line += f"  (n_train={m['n_train']}, n_test={m['n_test']})"
                print(line)
        except Exception as e:
            print(f"Per-city training skipped: {e}")
