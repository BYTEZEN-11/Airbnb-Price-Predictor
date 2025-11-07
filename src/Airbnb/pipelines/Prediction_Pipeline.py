import os
import sys

import pandas as pd

from Airbnb.exception import customexception
from Airbnb.logger import logging
from Airbnb.utils.utils import load_object

GLOBAL_PREPROC = os.path.join("Artifacts", "Preprocessor.pkl")
GLOBAL_MODEL   = os.path.join("Artifacts", "Model.pkl")
def _city_artifacts(city: str):
    if not city:
        return None
    city_dir = os.path.join("Artifacts", "Cities", city)
    preproc = os.path.join(city_dir, "Preprocessor.pkl")
    model   = os.path.join(city_dir, "Model.pkl")
    if not (os.path.exists(preproc) and os.path.exists(model)):
        return None
    return load_object(preproc), load_object(model)
class PredictPipeline:
    def __init__(self, city: str | None = None):
        self.city = city
        city_loaded = _city_artifacts(city) if city else None
        if city_loaded is not None:
            self._preprocessor, self._model = city_loaded
            logging.info(f"Loaded per-city model: {city}")
        else:
            self._preprocessor = load_object(GLOBAL_PREPROC)
            self._model        = load_object(GLOBAL_MODEL)
            logging.info("Loaded global model")
        self.city_used = city_loaded is not None
    def predict(self, features: pd.DataFrame):
        try:
            import numpy as np
            if not isinstance(features, pd.DataFrame):
                features = pd.DataFrame(features)
            try:
                scaled = self._preprocessor.transform(features)
                logging.info("Data Scaled")
                pred = self._model.predict(scaled)
            except ValueError as ve:
                arr = features.select_dtypes(include=[np.number]).values
                if arr.shape[1] != getattr(self._preprocessor, "n_features_in_", arr.shape[1]):
                    raise customexception(
                        ValueError(
                            f"Preprocessor expects {self._preprocessor.n_features_in_} "
                            f"features but received {arr.shape[1]}. "
                            f"Please retrain the model against the current feature schema."
                        ),
                        sys,
                    ) from ve
                scaled = self._preprocessor.transform(arr)
                pred = self._model.predict(scaled)
            try:
                from Airbnb.components.QuantilePredictor import predict_with_band
                point, lo, hi = predict_with_band(scaled, pred)
                return np.array([
                    float(np.atleast_1d(point)[0]),
                    float(np.atleast_1d(lo)[0]),
                    float(np.atleast_1d(hi)[0]),
                ])
            except Exception as e:
                logging.warning(f"Confidence band unavailable: {e}")
                return np.array([float(np.atleast_1d(pred)[0])] * 3)
        except Exception as e:
            logging.exception("predict pipeline failure")
            raise customexception(e, sys)
class CustomData:
    def __init__(self,
                 property_type: str,
                 room_type: str,
                 amenities: int,
                 accommodates: int,
                 bathrooms: float,
                 bed_type: str,
                 cancellation_policy: str,
                 cleaning_fee,
                 city: str,
                 host_has_profile_pic: str,
                 host_identity_verified: str,
                 host_response_rate: float,
                 instant_bookable: str,
                 latitude: float,
                 longitude: float,
                 number_of_reviews: int,
                 review_scores_rating: float,
                 bedrooms: int,
                 beds: int):
        self.property_type         = property_type
        self.room_type             = room_type
        self.amenities             = amenities
        self.accommodates          = accommodates
        self.bathrooms             = bathrooms
        self.bed_type              = bed_type
        self.cancellation_policy   = cancellation_policy
        self.cleaning_fee          = cleaning_fee
        self.city                  = city
        self.host_has_profile_pic  = host_has_profile_pic
        self.host_identity_verified = host_identity_verified
        self.host_response_rate    = host_response_rate
        self.instant_bookable      = instant_bookable
        self.latitude              = latitude
        self.longitude             = longitude
        self.number_of_reviews     = number_of_reviews
        self.review_scores_rating  = review_scores_rating
        self.bedrooms              = bedrooms
        self.beds                  = beds
    def get_data_as_dataframe(self) -> pd.DataFrame:
        try:
            data = {
                'property_type':         [self.property_type],
                'room_type':             [self.room_type],
                'amenities':             [self.amenities],
                'accommodates':          [self.accommodates],
                'bathrooms':             [self.bathrooms],
                'bed_type':              [self.bed_type],
                'cancellation_policy':   [self.cancellation_policy],
                'cleaning_fee':          [self.cleaning_fee],
                'city':                  [self.city],
                'host_has_profile_pic':  [self.host_has_profile_pic],
                'host_identity_verified':[self.host_identity_verified],
                'host_response_rate':    [self.host_response_rate],
                'instant_bookable':      [self.instant_bookable],
                'latitude':              [self.latitude],
                'longitude':             [self.longitude],
                'number_of_reviews':     [self.number_of_reviews],
                'review_scores_rating':  [self.review_scores_rating],
                'bedrooms':              [self.bedrooms],
                'beds':                  [self.beds],
            }
            df = pd.DataFrame(data)
            cat_cols = ['property_type','room_type','bed_type','cancellation_policy',
                        'city','host_has_profile_pic','host_identity_verified',
                        'instant_bookable']
            for c in cat_cols:
                df[c] = df[c].astype(object)
            df['cleaning_fee'] = df['cleaning_fee'].astype(object)
            logging.info('DataFrame assembled for prediction')
            return df
        except Exception as e:
            logging.exception('get_data_as_dataframe failure')
            raise customexception(e, sys)
