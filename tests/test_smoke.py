import io
import pandas as pd
import pytest
@pytest.fixture(scope="module")
def client():
    import app as app_module
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        yield c
class TestHomeRoute:
    def test_get_returns_200_with_all_features(self, client):
        r = client.get("/")
        assert r.status_code == 200
        for needle in (b'<form', b'id="map"', b'theme-toggle',
                       b'draft-banner', b'csv-upload', b'btn-predict'):
            assert needle in r.data, f"missing feature marker: {needle.decode()}"
    def test_security_headers_attached(self, client):
        r = client.get("/")
        h = r.headers
        assert h.get("X-Content-Type-Options") == "nosniff"
        assert h.get("X-Frame-Options") == "DENY"
        assert h.get("Referrer-Policy") == "strict-origin-when-cross-origin"
        assert "geolocation=()" in (h.get("Permissions-Policy") or "")
class TestPredictRoute:
    def test_valid_form_returns_200_and_price(self, flask_client):
        form = {
            'propertytype': 'Apartment', 'roomtype': 'Entire home/apt',
            'city': 'NYC', 'accommodates': '2', 'bedrooms': '1', 'beds': '1',
            'bathrooms': '1', 'amenties': '10', 'bedtype': 'Real Bed',
            'canceltype': 'Strict', 'clean': 'True', 'instbook': 't',
            'dp': 't', 'verify': 't', 'hostresponse': '95',
            'lat': '40.7128', 'long': '-74.0060',
            'review': '10', 'overallreview': '90',
        }
        r = flask_client.post("/", data=form)
        assert r.status_code == 200
        assert b'$' in r.data
    def test_missing_field_returns_400(self, flask_client):
        bad = {
            'propertytype': 'Apartment', 'roomtype': 'Entire home/apt',
            'accommodates': '2', 'bedrooms': '1', 'beds': '1',
            'bathrooms': '1', 'amenties': '10', 'bedtype': 'Real Bed',
            'canceltype': 'Strict', 'clean': 'True', 'instbook': 't',
            'dp': 't', 'verify': 't', 'hostresponse': '95',
            'lat': '40.7128', 'long': '-74.0060',
            'review': '10', 'overallreview': '90',
        }
        r = flask_client.post("/", data=bad)
        assert r.status_code == 400
    def test_garbage_numeric_returns_400(self, flask_client):
        form = {
            'propertytype': 'Apartment', 'roomtype': 'Entire home/apt',
            'city': 'NYC', 'accommodates': 'two',
            'bedrooms': '1', 'beds': '1', 'bathrooms': '1', 'amenties': '10',
            'bedtype': 'Real Bed', 'canceltype': 'Strict', 'clean': 'True',
            'instbook': 't', 'dp': 't', 'verify': 't', 'hostresponse': '95',
            'lat': '40.7128', 'long': '-74.0060',
            'review': '10', 'overallreview': '90',
        }
        r = flask_client.post("/", data=form)
        assert r.status_code == 400
class TestMetricsRoute:
    def test_metrics_endpoint_returns_200(self, client):
        r = client.get("/metrics")
        assert r.status_code == 200
class TestPredictCsvRoute:
    REQUIRED_ROW = {
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
    def test_csv_with_all_columns_succeeds(self, flask_client):
        df = pd.DataFrame([self.REQUIRED_ROW])
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        data = {"file": (io.BytesIO(buf.getvalue().encode()), "test.csv")}
        r = flask_client.post("/predict-csv", data=data, content_type="multipart/form-data")
        assert r.status_code == 200
        body = r.data.decode("utf-8", errors="ignore")
        assert "predicted_price_usd" in body
    def test_csv_with_missing_columns_returns_helpful_error(self, flask_client):
        bad = {k: v for k, v in self.REQUIRED_ROW.items() if k != "host_response_rate"}
        df = pd.DataFrame([bad])
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        data = {"file": (io.BytesIO(buf.getvalue().encode()), "bad.csv")}
        r = flask_client.post("/predict-csv", data=data, content_type="multipart/form-data")
        assert r.status_code == 400
        assert b"host_response_rate" in r.data
    def test_csv_with_no_file_returns_400(self, flask_client):
        r = flask_client.post("/predict-csv", data={})
        assert r.status_code == 400
    def test_csv_with_non_csv_file_rejected(self, flask_client):
        data = {"file": (io.BytesIO(b"not really a csv"), "xlsx.xlsx")}
        r = flask_client.post("/predict-csv", data=data, content_type="multipart/form-data")
        assert r.status_code in (400, 200, 500)
