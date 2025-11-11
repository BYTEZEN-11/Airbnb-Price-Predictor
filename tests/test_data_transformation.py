import numpy as np
import pandas as pd
import pytest
from Airbnb.components.Data_transformation import (
    count_amenities,
    parse_host_response_rate,
    vectorize_amenity_count,
)
class TestParseHostResponseRate:
    def test_strips_percent_and_returns_float(self):
        s = pd.Series(["100%", "0%", "95%"])
        out = parse_host_response_rate(s)
        assert out.dtype.kind == "f"
        assert out.tolist() == pytest.approx([100.0, 0.0, 95.0])
    def test_handles_missing_values(self):
        s = pd.Series(["100%", None, np.nan, "0%"])
        out = parse_host_response_rate(s)
        assert out.iloc[0] == 100.0
        assert out.iloc[3] == 0.0
        assert pd.isna(out.iloc[1])
        assert pd.isna(out.iloc[2])
    def test_preserves_length(self):
        s = pd.Series(["50%"] * 1000)
        assert len(parse_host_response_rate(s)) == 1000
    def test_no_index_misalignment(self):
        s = pd.Series(["80%", None, "60%", np.nan, "100%"])
        out = parse_host_response_rate(s)
        assert out.iloc[0] == 80.0
        assert out.iloc[2] == 60.0
        assert out.iloc[4] == 100.0
class TestCountAmenities:
    def test_brace_set(self):
        assert count_amenities('{"Wifi","AC",Kitchen,Heating}') == 4
    def test_empty_brace_set(self):
        assert count_amenities("{}") == 0
    def test_comma_delimited_fallback(self):
        assert count_amenities("Wifi,Kitchen,Heating") == 3
    def test_nan_returns_zero(self):
        assert count_amenities(None) == 0
        assert count_amenities(float("nan")) == 0
    def test_unclosed_brace_does_not_crash(self):
        result = count_amenities("{not valid json")
        assert result in (0, 1)
    def test_garbage_with_outer_braces_returns_zero(self):
        assert count_amenities("{}") == 0
class TestVectorizeAmenityCount:
    def test_elementwise_application(self):
        s = pd.Series([
            '{"Wifi","AC",Heating}',
            "{}",
            np.nan,
            "Single",
        ])
        out = vectorize_amenity_count(s)
        assert out.tolist() == [3, 0, 0, 1]
