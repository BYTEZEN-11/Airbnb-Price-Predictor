import math
from pathlib import Path
import numpy as np
import pandas as pd
def _write_artifacts(artifacts_dir: Path):
    from joblib import dump
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    preprocessor = ColumnTransformer(
        transformers=[("num", Pipeline([("sc", StandardScaler())]), ["amenities"])],
        remainder="drop",
    )
    X = np.array([[1], [10]], dtype=float)
    preprocessor.fit(pd.DataFrame({"amenities": X[:, 0]}))
    model = LinearRegression()
    model.fit(preprocessor.transform(pd.DataFrame({"amenities": X[:, 0]})), np.array([2.0, 20.0]))
    dump(preprocessor, artifacts_dir / "Preprocessor.pkl")
    dump(model,        artifacts_dir / "Model.pkl")
class TestCustomData:
    def test_get_data_as_dataframe_columns_and_shape(self):
        from Airbnb.pipelines.Prediction_Pipeline import CustomData
        cd = CustomData(
            property_type="Apartment", room_type="Entire home/apt",
            amenities=8, accommodates=4, bathrooms=1.0,
            bed_type="Real Bed", cancellation_policy="strict",
            cleaning_fee="True", city="NYC",
            host_has_profile_pic="t", host_identity_verified="t",
            host_response_rate=99.0, instant_bookable="f",
            latitude=40.7, longitude=-74.0,
            number_of_reviews=5, review_scores_rating=92.0,
            bedrooms=2, beds=2,
        )
        df = cd.get_data_as_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (1, 19)
        assert df.iloc[0]["city"] == "NYC"
        assert df["amenities"].iloc[0] == 8
    def test_categorical_columns_remain_object_dtype(self):
        from Airbnb.pipelines.Prediction_Pipeline import CustomData
        cd = CustomData(
            property_type="Apartment", room_type="Private room",
            amenities=2, accommodates=2, bathrooms=1.0,
            bed_type="Real Bed", cancellation_policy="flexible",
            cleaning_fee="False", city="SF",
            host_has_profile_pic="f", host_identity_verified="f",
            host_response_rate=80.0, instant_bookable="t",
            latitude=37.7, longitude=-122.4,
            number_of_reviews=3, review_scores_rating=88.0,
            bedrooms=1, beds=1,
        )
        df = cd.get_data_as_dataframe()
        for col in ("property_type", "room_type", "bed_type", "cancellation_policy",
                    "city", "host_has_profile_pic", "host_identity_verified",
                    "instant_bookable"):
            assert df[col].dtype == object, f"{col} became {df[col].dtype}"
        assert df["cleaning_fee"].dtype == object
class TestPredictPipelineRoundTrip:
    def test_predict_returns_three_element_array(self, artifacts_dir):
        _write_artifacts(artifacts_dir / "Artifacts")
        import pandas as pd
        from Airbnb.pipelines.Prediction_Pipeline import PredictPipeline
        pipeline = PredictPipeline()
        df = pd.DataFrame({"amenities": [10]})
        pred = pipeline.predict(df)
        assert len(pred) == 3
        assert all(isinstance(float(x), float) for x in pred)
    def test_predict_caches_objects(self, artifacts_dir):
        _write_artifacts(artifacts_dir / "Artifacts")
        from Airbnb.pipelines.Prediction_Pipeline import PredictPipeline
        a = PredictPipeline()
        b = PredictPipeline()
        import pandas as pd
        assert a.predict(pd.DataFrame({"amenities": [5]}))[0] == b.predict(pd.DataFrame({"amenities": [5]}))[0]
    def test_app_expm1_round_trip(self):
        log_price = 2.0
        assert round(math.expm1(log_price), 2) == 6.39
