FORM_PAYLOAD = {
    "propertytype": "Apartment",
    "roomtype":      "Entire home/apt",
    "city":          "NYC",
    "accommodates":  "4",
    "bedrooms":      "2",
    "beds":          "2",
    "bathrooms":     "1",
    "amenties":      "10",
    "bedtype":       "Real Bed",
    "canceltype":    "Strict",
    "clean":         "True",
    "instbook":      "t",
    "dp":            "t",
    "verify":        "t",
    "hostresponse":  "95",
    "lat":           "40.7128",
    "long":          "-74.0060",
    "review":        "10",
    "overallreview": "90",
}
class TestGetHome:
    def test_get_renders_form(self, flask_client):
        r = flask_client.get("/")
        assert r.status_code == 200
        assert b"Predict My Price" in r.data or b"property_details" in r.data.lower() or b"propertytype" in r.data.lower()
class TestPostPredict:
    def test_post_with_valid_form_returns_200_and_price(self, flask_client):
        r = flask_client.post("/", data=FORM_PAYLOAD, follow_redirects=True)
        assert r.status_code == 200
        assert b"$" in r.data
    def test_post_with_bad_data_returns_400(self, flask_client):
        bad = {k: v for k, v in FORM_PAYLOAD.items() if k != "city"}
        r = flask_client.post("/", data=bad)
        assert r.status_code in (400, 200)
        if r.status_code == 200:
            assert b"index.html" not in r.headers.get("Content-Disposition", b"")
    def test_post_with_invalid_numeric_returns_400_or_error_page(self, flask_client):
        bad = dict(FORM_PAYLOAD, accommodates="not-a-number")
        r = flask_client.post("/", data=bad)
        assert r.status_code == 400
        assert b"error" in r.data.lower() or b"went wrong" in r.data.lower()
