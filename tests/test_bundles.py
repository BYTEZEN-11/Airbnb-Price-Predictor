import io
import numpy as np
import pandas as pd
from Airbnb.components.Data_transformation import (
    add_seasonality_features,
)
class TestSeasonality:
    def test_first_review_month_quarter(self):
        df = pd.DataFrame({"first_review": ["2018-06-12", "2020-11-25", "bad", None]})
        out = add_seasonality_features(df)
        assert out["first_review_month"].iloc[0] == 6
        assert out["first_review_quarter"].iloc[1] == 4
        assert pd.isna(out["first_review_month"].iloc[2])
        assert pd.isna(out["first_review_month"].iloc[3])
    def test_holiday_season_flag(self):
        df = pd.DataFrame({"first_review": ["2020-11-25", "2020-07-04"]})
        out = add_seasonality_features(df)
        assert int(out["is_holiday_season"].iloc[0]) == 1
        assert int(out["is_holiday_season"].iloc[1]) == 0
    def test_host_tenure(self):
        df = pd.DataFrame({"host_since": ["2010-01-01", None]})
        out = add_seasonality_features(df)
        assert out["host_tenure_days"].iloc[0] > 3000
        assert pd.isna(out["host_tenure_days"].iloc[1])
    def test_days_since_first_review(self):
        df = pd.DataFrame({"first_review": ["2020-01-01"]})
        out = add_seasonality_features(df)
        assert out["days_since_first_review"].iloc[0] > 2000
from Airbnb.components.QuantilePredictor import predict_with_band
class TestConfidenceBand:
    def test_envelope_fallback_when_no_models(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "Artifacts").mkdir()
        pt, lo, hi = predict_with_band(features_scaled=None, main_pred=0.0)
        assert pt == 0.0
        assert abs(lo - (-0.15)) < 1e-6
        assert abs(hi -   0.15) < 1e-6
class TestPerCityTrainer:
    def _toy_data(self):
        cats = ['property_type', 'room_type', 'bed_type', 'cancellation_policy',
                'cleaning_fee', 'city', 'host_identity_verified',
                'instant_bookable', 'host_has_profile_pic']
        rows = []
        for city in ["NYC", "SF", "LA", "Chicago", "Boston", "DC"]:
            for i in range(60):
                row = {
                    "city": city,
                    "log_price": 5.0 + i * 0.01,
                    "id": i,
                    "first_review": "2020-01-01",
                    "host_since":   "2015-06-15",
                    "host_response_rate": "95%",
                    "amenities": '{"Wifi","AC"}',
                    "name": "x", "description": "x", "last_review": "2020-01-01",
                    "neighbourhood": "x", "thumbnail_url": "x", "zipcode": "00000",
                    "accommodates": i % 6 + 1,
                    "bathrooms": 1.0,
                    "bedrooms": 1.0,
                    "beds": 1.0,
                    "latitude": 40.7, "longitude": -74.0,
                    "number_of_reviews": i,
                    "review_scores_rating": 90.0,
                }
                for c in cats:
                    defaults = {
                        'property_type':'Apartment','room_type':'Entire home/apt',
                        'bed_type':'Real Bed','cancellation_policy':'strict',
                        'cleaning_fee':'True','city':city,
                        'host_identity_verified':'t','instant_bookable':'f',
                        'host_has_profile_pic':'t'
                    }
                    row[c] = defaults[c]
                rows.append(row)
        return pd.DataFrame(rows)
    def test_all_cities_recorded(self, artifacts_dir):
        from Airbnb.components.PerCityTrainer import train_per_city
        df = self._toy_data()
        train_idx, test_idx = [], []
        for city in df["city"].unique():
            mask = df["city"] == city
            city_idx = df.index[mask].tolist()
            n_test = max(2, len(city_idx) // 5)
            test_idx.extend(city_idx[:n_test])
            train_idx.extend(city_idx[n_test:])
        train = df.loc[train_idx].reset_index(drop=True)
        test  = df.loc[test_idx ].reset_index(drop=True)
        import Airbnb.components.PerCityTrainer as mod
        old = mod.MIN_ROWS_PER_CITY
        mod.MIN_ROWS_PER_CITY = 5
        try:
            metrics = train_per_city(train, test)
        finally:
            mod.MIN_ROWS_PER_CITY = old
        assert isinstance(metrics, dict)
        for city in ("NYC", "SF", "LA", "Chicago", "Boston", "DC"):
            if city in metrics:
                assert metrics[city]["status"] in ("trained", "fallback-global")
class TestCsvUploadRoute:
    MINIMAL_ROW = {
        "id": "1", "log_price": "5",
        "property_type": "Apartment", "room_type": "Entire home/apt",
        "amenities": '{"Wifi","AC"}', "accommodates": "2",
        "bathrooms": "1.0", "bed_type": "Real Bed",
        "cancellation_policy": "strict", "cleaning_fee": "True",
        "city": "NYC",
        "description": "", "first_review": "2020-01-01",
        "host_has_profile_pic": "t", "host_identity_verified": "t",
        "host_response_rate": "95%",
        "host_since": "2015-06-15",
        "instant_bookable": "f", "last_review": "2020-01-01",
        "latitude": "40.7128", "longitude": "-74.0060", "name": "",
        "neighbourhood": "", "number_of_reviews": "3",
        "review_scores_rating": "90", "thumbnail_url": "", "zipcode": "",
        "bedrooms": "1", "beds": "1",
    }
    def test_csv_upload_returns_csv(self, flask_client):
        df = pd.DataFrame([self.MINIMAL_ROW])
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        data = {"file": (io.BytesIO(buf.getvalue().encode()), "test.csv")}
        r = flask_client.post("/predict-csv", data=data, content_type="multipart/form-data")
        assert r.status_code == 200
        assert r.headers.get("Content-Type", "").startswith("text/csv")
        body = r.data.decode("utf-8", errors="ignore")
        assert "predicted_price_usd" in body
    def test_csv_upload_no_file_400(self, flask_client):
        r = flask_client.post("/predict-csv", data={})
        assert r.status_code == 400
    def test_csv_upload_garbage_400(self, flask_client):
        data = {"file": (io.BytesIO(b"not,a,csv\njust,text"), "bad.csv")}
        r = flask_client.post("/predict-csv", data=data, content_type="multipart/form-data")
        assert r.status_code in (400, 500)
class TestMetricsRoute:
    def test_metrics_endpoint_returns_200(self, flask_client):
        r = flask_client.get("/metrics")
        assert r.status_code == 200
class TestOptuna:
    def test_optuna_handles_missing_dependency(self, monkeypatch):
        import builtins
        import importlib
        import sys
        real_import = builtins.__import__
        def fake_import(name, *args, **kwargs):
            if name == "optuna" or name.startswith("optuna."):
                raise ImportError("optuna hidden for test")
            return real_import(name, *args, **kwargs)
        monkeypatch.setattr(builtins, "__import__", fake_import)
        if "Airbnb.components.HyperparameterTuner" in sys.modules:
            importlib.reload(sys.modules["Airbnb.components.HyperparameterTuner"])
        else:
            import Airbnb.components.HyperparameterTuner
            importlib.reload(
                sys.modules["Airbnb.components.HyperparameterTuner"]
            )
        from Airbnb.components.HyperparameterTuner import run_optuna_search
        out = run_optuna_search(np.zeros((5, 3)), np.zeros(5),
                                np.zeros((2, 3)), np.zeros(2), n_trials=1)
        assert out == (None, None, None)
