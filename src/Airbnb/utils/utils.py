import hashlib
import os
import sys

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from Airbnb.exception import customexception
from Airbnb.logger import logging

MANIFEST_PATH = os.path.join("Artifacts", "manifest.sha256")
def _compute_sha256(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
def write_manifest(file_paths):
    digests = {os.path.basename(p): _compute_sha256(p) for p in file_paths if os.path.exists(p)}
    with open(MANIFEST_PATH, "w") as f:
        f.writelines(f"{digest}  {name}\n" for name, digest in digests.items())
    logging.info(f"Wrote manifest at {MANIFEST_PATH}")
def _verify_against_manifest(file_path: str):
    if not os.path.exists(MANIFEST_PATH):
        return
    name = os.path.basename(file_path)
    expected = None
    with open(MANIFEST_PATH) as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) == 2 and parts[1] == name:
                expected = parts[0]
                break
    if expected is None:
        logging.warning(f"No manifest entry for {name}; skipping verification")
        return
    actual = _compute_sha256(file_path)
    if actual != expected:
        raise customexception(
            ValueError(f"Integrity check failed for {name}: expected {expected}, got {actual}"),
            sys,
        )
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            joblib.dump(obj, file_obj, compress=3)
        write_manifest([file_path])
    except (IOError, OSError, ValueError) as e:
        raise customexception(e, sys)
def load_object(file_path):
    try:
        _verify_against_manifest(file_path)
        with open(file_path, "rb") as file_obj:
            return joblib.load(file_obj)
    except (IOError, OSError, ValueError) as e:
        raise customexception(e, sys)
def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        for i in range(len(models)):
            name = list(models.keys())[i]
            model = list(models.values())[i]
            model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)
            r2  = r2_score(y_test, y_test_pred)
            rmse = float(np.sqrt(mean_squared_error(y_test, y_test_pred)))
            mae  = float(mean_absolute_error(y_test, y_test_pred))
            report[name] = {"r2": r2, "rmse": rmse, "mae": mae}
        return report
    except (ValueError, TypeError, AttributeError) as e:
        logging.exception("evaluate_model failure")
        raise customexception(e, sys)
