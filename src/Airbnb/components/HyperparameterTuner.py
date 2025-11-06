from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from Airbnb.logger import logging
from Airbnb.utils.utils import save_object


@dataclass
class TunerConfig:
    best_params_path: str = os.path.join("Artifacts", "best_params.json")
    best_model_path:   str = os.path.join("Artifacts", "Model.pkl")
    n_trials: int = 25
    enable_mlflow: bool = False
def _build_catboost(trial):
    from catboost import CatBoostRegressor
    return CatBoostRegressor(
        iterations=trial.suggest_int("it", 200, 800),
        depth=trial.suggest_int("depth", 4, 8),
        learning_rate=trial.suggest_float("lr", 0.01, 0.2, log=True),
        l2_leaf_reg=trial.suggest_float("l2", 1e-2, 10.0, log=True),
        verbose=False,
        allow_writing_files=False,
        random_state=42,
    )
def _build_xgboost(trial):
    from xgboost import XGBRegressor
    return XGBRegressor(
        n_estimators=trial.suggest_int("n", 200, 800),
        max_depth=trial.suggest_int("depth", 4, 8),
        learning_rate=trial.suggest_float("lr", 0.01, 0.2, log=True),
        subsample=trial.suggest_float("sub", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("csb", 0.6, 1.0),
        n_jobs=-1,
        random_state=42,
    )
def _build_lightgbm(trial):
    try:
        import lightgbm as lgb
    except ImportError:
        return None
    return lgb.LGBMRegressor(
        n_estimators=trial.suggest_int("n", 200, 800),
        num_leaves=trial.suggest_int("leaves", 16, 128),
        learning_rate=trial.suggest_float("lr", 0.01, 0.2, log=True),
        subsample=trial.suggest_float("sub", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("csb", 0.6, 1.0),
        n_jobs=-1,
        random_state=42,
    )
SEARCH_SPACES = [
    ("CatBoost", _build_catboost),
    ("XGBoost",  _build_xgboost),
    ("LightGBM", _build_lightgbm),
]
def _score(model, X_train, y_train, X_test, y_test) -> float:
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return float(np.sqrt(np.mean((pred - y_test) ** 2)))
def run_optuna_search(X_train, y_train, X_test, y_test, n_trials: int = 25):
    try:
        import optuna
    except ImportError:
        logging.warning("optuna not installed; skipping search.")
        return None, None, None
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    try:
        import mlflow
        if os.environ.get("AIRBNB_ENABLE_MLFLOW") == "1":
            mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", ""))
            mlflow.set_experiment("Airbnb-Price")
            use_mlflow = True
        else:
            use_mlflow = False
    except (ImportError, AttributeError, RuntimeError):
        use_mlflow = False
    best = {"name": None, "model": None, "params": None, "rmse": float("inf")}
    for model_name, factory in SEARCH_SPACES:
        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        def objective(trial, _model_name=model_name, _factory=factory):
            model = _factory(trial)
            if model is None:
                raise optuna.exceptions.TrialPruned()
            try:
                rmse = _score(model, X_train, y_train, X_test, y_test)
            except (ValueError, TypeError, RuntimeError) as e:
                logging.warning(f"{_model_name} trial failed: {e}")
                raise optuna.exceptions.TrialPruned()
            if use_mlflow:
                with mlflow.start_run(nested=True):
                    mlflow.log_metric("rmse", rmse)
                    mlflow.log_param("model", _model_name)
                    mlflow.log_params(trial.params)
            return rmse
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        if len(study.trials) == 0:
            continue
        best_trial = study.best_trial
        if best_trial.value < best["rmse"]:
            best_model = factory(best_trial)
            best_model.fit(X_train, y_train)
            best.update({
                "name": model_name,
                "model": best_model,
                "params": {"model": model_name, **best_trial.params},
                "rmse": best_trial.value,
            })
    if best["model"] is None:
        return None, None, None
    import json
    with open(TunerConfig.best_params_path, "w") as f:
        json.dump(best["params"], f, indent=2)
    save_object(TunerConfig.best_model_path, best["model"])
    logging.info(
        f"Optuna best: {best['name']} rmse={best['rmse']:.4f} params={best['params']}"
    )
    return best["name"], best["model"], best["params"]
