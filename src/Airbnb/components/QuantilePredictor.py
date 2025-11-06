from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from catboost import CatBoostRegressor

from Airbnb.utils.utils import load_object, save_object


@dataclass
class QuantileConfig:
    q10_path: str = os.path.join("Artifacts", "Model_q10.pkl")
    q90_path: str = os.path.join("Artifacts", "Model_q90.pkl")
    q10_alpha: float = 0.10
    q90_alpha: float = 0.90
def train_quantiles(X_train, y_train) -> Tuple[object, object]:
    q10 = CatBoostRegressor(
        loss_function="Quantile:alpha=0.10",
        iterations=300, depth=6, learning_rate=0.05,
        verbose=False, allow_writing_files=False, random_state=42,
    ).fit(X_train, y_train)
    q90 = CatBoostRegressor(
        loss_function="Quantile:alpha=0.90",
        iterations=300, depth=6, learning_rate=0.05,
        verbose=False, allow_writing_files=False, random_state=42,
    ).fit(X_train, y_train)
    save_object(QuantileConfig.q10_path, q10)
    save_object(QuantileConfig.q90_path, q90)
    return q10, q90
def predict_with_band(features_scaled, main_pred: float | np.ndarray):
    try:
        q10 = load_object(QuantileConfig.q10_path)
        q90 = load_object(QuantileConfig.q90_path)
        if np.ndim(main_pred) == 0:
            lo = float(q10.predict(features_scaled)[0])
            hi = float(q90.predict(features_scaled)[0])
            return float(main_pred), lo, hi
        else:
            lo = q10.predict(features_scaled)
            hi = q90.predict(features_scaled)
            return np.asarray(main_pred), lo, hi
    except Exception:
        if np.ndim(main_pred) == 0:
            return float(main_pred), float(main_pred) - 0.15, float(main_pred) + 0.15
        arr = np.asarray(main_pred)
        return arr, arr - 0.15, arr + 0.15
