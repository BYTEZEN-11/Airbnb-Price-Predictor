import ast
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from Airbnb.exception import customexception
from Airbnb.logger import logging
from Airbnb.utils.cities import (
    BED_TYPE_CAT,
    CANCELLATION_CAT,
    CANONICAL_CITIES,
    CATEGORICAL_COLS,
    CLEANING_FEE_CAT,
    NUMERICAL_COLS,
    PROPERTY_TYPE_CAT,
    ROOM_TYPE_CAT,
)
from Airbnb.utils.utils import save_object


def _reference_date() -> pd.Timestamp:
    env = os.environ.get("AIRBNB_REFERENCE_DATE")
    if env:
        return pd.Timestamp(env)
    return pd.Timestamp("2026-01-01")
def parse_host_response_rate(series: pd.Series) -> pd.Series:
    raw = series.astype(object)
    raw = raw.where(raw.notna() & (raw.astype(str).str.strip() != ''), np.nan)
    return (
        raw.astype(str)
            .str.rstrip('%')
            .replace('nan', np.nan)
            .astype(float)
    )
def _split_top_level(s: str, sep: str = ",") -> list[str]:
    out, depth, in_quote = [], 0, None
    i = 0
    while i < len(s):
        c = s[i]
        if in_quote:
            if c == "\\" and i + 1 < len(s):
                i += 2
                continue
            if c == in_quote:
                in_quote = None
        else:
            if c in ('"', "'"):
                in_quote = c
            elif c in "{[(":
                depth += 1
            elif c in "}])":
                depth -= 1
            elif c == sep and depth == 0:
                out.append(s[:i])
                s = s[i+1:]
                i = 0
                continue
        i += 1
    if s:
        out.append(s)
    return out
def count_amenities(val) -> int:
    if pd.isna(val):
        return 0
    s = str(val).strip()
    if not s or s in ("{}", "[]"):
        return 0
    inner = s[1:-1].strip() if (len(s) >= 2 and s[0] in "{[" and s[-1] in "}]") else s
    if not inner:
        return 0
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (set, list, tuple)):
            return len(parsed)
    except (ValueError, SyntaxError):
        pass
    return sum(1 for item in _split_top_level(inner, ",") if item.strip())
def vectorize_amenity_count(series: pd.Series) -> pd.Series:
    return series.apply(count_amenities)
def parse_date_column(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors='coerce')
def add_seasonality_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'first_review' in df.columns:
        fr = parse_date_column(df['first_review'])
        ref = _reference_date()
        df['first_review_month']  = fr.dt.month
        df['first_review_quarter'] = fr.dt.quarter
        df['days_since_first_review'] = (ref - fr).dt.days
    if 'host_since' in df.columns:
        hs = parse_date_column(df['host_since'])
        ref = _reference_date()
        df['host_tenure_days'] = (ref - hs).dt.days
    if 'first_review_month' in df.columns:
        df['is_holiday_season'] = df['first_review_month'].isin([11, 12]).astype(int)
    return df
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('Artifacts', 'Preprocessor.pkl')
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    def get_data_transformation(self):
        try:
            logging.info('Data Transformation initiated')
            numerical_cols = NUMERICAL_COLS
            categorical_cols = CATEGORICAL_COLS
            property_type_cat = PROPERTY_TYPE_CAT
            room_type_cat = ROOM_TYPE_CAT
            bed_type_cat = BED_TYPE_CAT
            cancellation_policy_cat = CANCELLATION_CAT
            cleaning_fee_cat = CLEANING_FEE_CAT
            city_cat = CANONICAL_CITIES
            host_has_profile_pic_cat = ['t','f']
            host_identity_verified_cat = ['t','f']
            instant_bookable_cat = ['t','f']
            logging.info('Pipeline Initiated')
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler',  StandardScaler()),
            ])
            cat_pipeline = Pipeline(steps=[
                ('imputer',        SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder', OrdinalEncoder(
                    categories=[
                        property_type_cat, room_type_cat, bed_type_cat,
                        cancellation_policy_cat, cleaning_fee_cat, city_cat,
                        host_has_profile_pic_cat, host_identity_verified_cat,
                        instant_bookable_cat,
                    ],
                    handle_unknown='use_encoded_value',
                    unknown_value=-1,
                )),
                ('scaler', StandardScaler()),
            ])
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, numerical_cols),
                    ('cat_pipeline', cat_pipeline, categorical_cols),
                ],
                remainder='drop',
                verbose_feature_names_out=False,
            )
            return preprocessor
        except Exception as e:
            logging.info("Exception occurred in get_data_transformation")
            raise customexception(e, sys)
    def initialize_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df  = pd.read_csv(test_path)
            logging.info("read train and test data complete")
            preprocessing_obj = self.get_data_transformation()
            train_df['host_response_rate'] = parse_host_response_rate(train_df['host_response_rate'])
            test_df['host_response_rate']  = parse_host_response_rate(test_df['host_response_rate'])
            logging.info("Host Response Rate converted to float (0-100)")
            train_df['amenities'] = vectorize_amenity_count(train_df['amenities'])
            test_df['amenities']  = vectorize_amenity_count(test_df['amenities'])
            train_df = add_seasonality_features(train_df)
            test_df  = add_seasonality_features(test_df)
            target_column_name = 'log_price'
            drop_columns = [
                target_column_name, 'id', 'name', 'description', 'first_review',
                'host_since', 'last_review', 'neighbourhood', 'thumbnail_url', 'zipcode'
            ]
            input_feature_train_df  = train_df.drop(columns=drop_columns)
            target_feature_train_df = train_df[target_column_name]
            input_feature_test_df  = test_df.drop(columns=drop_columns)
            target_feature_test_df = test_df[target_column_name]
            logging.info(f'Input Feature Train Dataframe Head : \n{input_feature_train_df.head().to_string()}')
            logging.info(f'Target Feature Train Dataframe Head : \n{target_feature_train_df.head().to_string()}')
            logging.info(f'{input_feature_train_df.dtypes}')
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr  = preprocessing_obj.transform(input_feature_test_df)
            logging.info("Applying preprocessing object on training and testing datasets.")
            train_arr = np.concatenate([input_feature_train_arr, np.array(target_feature_train_df).reshape(-1, 1)], axis=1)
            test_arr  = np.concatenate([input_feature_test_arr,  np.array(target_feature_test_df).reshape(-1, 1)], axis=1)
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj,
            )
            logging.info("preprocessing pickle file saved")
            return (train_arr, test_arr)
        except Exception as e:
            logging.info("Exception occurred in initialize_data_transformation")
            raise customexception(e, sys)
