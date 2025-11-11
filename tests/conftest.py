from __future__ import annotations
import sys
from pathlib import Path
import pytest
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))
@pytest.fixture
def artifacts_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "Artifacts").mkdir()
    return tmp_path
@pytest.fixture
def flask_client(artifacts_dir):
    import numpy as np
    import pandas as pd
    from joblib import dump
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    preprocessor = ColumnTransformer(
        transformers=[("num", Pipeline([("sc", StandardScaler())]), ["amenities"])],
        remainder="drop",
    )
    X = np.array([[1],[2],[3],[4],[5]], dtype=float)
    y = np.array([2, 4, 6, 8, 10], dtype=float)
    preprocessor.fit(pd.DataFrame({"amenities": X[:, 0]}))
    model = LinearRegression().fit(preprocessor.transform(pd.DataFrame({"amenities": X[:, 0]})), y)
    dump(preprocessor, artifacts_dir / "Artifacts" / "Preprocessor.pkl")
    dump(model,        artifacts_dir / "Artifacts" / "Model.pkl")
    from app import app
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client
