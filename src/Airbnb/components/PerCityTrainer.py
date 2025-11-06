from __future__ import annotations

import json
import os

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from Airbnb.components.Data_transformation import (
    add_seasonality_features,
    parse_host_response_rate,
    vectorize_amenity_count,
)
from Airbnb.logger import logging
from Airbnb.utils.cities import (
    BED_TYPE_CAT,
    BOOLEAN_CAT,
    CANCELLATION_CAT,
    CANONICAL_CITIES,
    CATEGORICAL_COLS,
    CLEANING_FEE_CAT,
    NUMERICAL_COLS,
    PROPERTY_TYPE_CAT,
    ROOM_TYPE_CAT,
)
from Airbnb.utils.utils import save_object

CITY_DIR = os.path.join("Artifacts", "Cities")
METRICS_PATH = os.path.join("Artifacts", "city_metrics.json")
MIN_ROWS_PER_CITY = 5_000
def _make_preprocessor(cities_seen: list[str]) -> ColumnTransformer:
    cities_for_encoder = [c for c in CANONICAL_CITIES if c in cities_seen] or cities_seen
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            categories=[
                PROPERTY_TYPE_CAT,
                ROOM_TYPE_CAT,
                BED_TYPE_CAT,
                CANCELLATION_CAT,
                CLEANING_FEE_CAT,
                cities_for_encoder,
                BOOLEAN_CAT,
                BOOLEAN_CAT,
                BOOLEAN_CAT,
            ],
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
        ("scaler", StandardScaler()),
    ])
    return ColumnTransformer([
        ("num", num_pipeline, NUMERICAL_COLS),
        ("cat", cat_pipeline, CATEGORICAL_COLS),
    ])
def train_per_city(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str = "log_price",
    drop_cols: list[str] | None = None,
) -> dict:
    if drop_cols is None:
        drop_cols = ['id','name','description','first_review','host_since',
                     'last_review','neighbourhood','thumbnail_url','zipcode']
    df_train = train_df.copy()
    df_test = test_df.copy()
    df_train['host_response_rate'] = parse_host_response_rate(df_train['host_response_rate'])
    df_test['host_response_rate']  = parse_host_response_rate(df_test['host_response_rate'])
    df_train['amenities'] = vectorize_amenity_count(df_train['amenities'])
    df_test['amenities']  = vectorize_amenity_count(df_test['amenities'])
    df_train = add_seasonality_features(df_train)
    df_test  = add_seasonality_features(df_test)
    metrics: dict = {}
    os.makedirs(CITY_DIR, exist_ok=True)
    for city in CANONICAL_CITIES:
        train_city = df_train[df_train['city'] == city].reset_index(drop=True)
        test_city  = df_test [df_test ['city'] == city].reset_index(drop=True)
        n_train, n_test = len(train_city), len(test_city)
        if n_train < MIN_ROWS_PER_CITY or n_test == 0:
            metrics[city] = {"status": "fallback-global", "n_train": n_train, "n_test": n_test}
            logging.info(f"[{city}] {n_train} rows (< {MIN_ROWS_PER_CITY} train or empty test); using global fallback")
            continue
        X_tr = train_city.drop(columns=[target] + drop_cols, errors='ignore')
        y_tr = train_city[target]
        X_te = test_city.drop(columns=[target] + drop_cols, errors='ignore')
        y_te = test_city[target]
        preprocessor = _make_preprocessor(df_train['city'].unique().tolist())
        X_tr_arr = preprocessor.fit_transform(X_tr)
        X_te_arr = preprocessor.transform(X_te)
        model = LinearRegression()
        model.fit(X_tr_arr, y_tr)
        r2 = float(model.score(X_te_arr, y_te))
        city_path = os.path.join(CITY_DIR, city)
        os.makedirs(city_path, exist_ok=True)
        save_object(os.path.join(city_path, "Preprocessor.pkl"), preprocessor)
        save_object(os.path.join(city_path, "Model.pkl"), model)
        metrics[city] = {
            "status": "trained",
            "n_train": n_train,
            "n_test":  n_test,
            "r2":      r2,
        }
        logging.info(f"[{city}] trained r2={r2:.4f} (n_train={n_train}, n_test={n_test})")
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics
