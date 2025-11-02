import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler

LOG_DIR = os.environ.get("AIRBNB_LOG_DIR", "logs")
def configure():
    root = logging.getLogger()
    if getattr(root, "_airbnb_configured", False):
        return
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    fh = RotatingFileHandler(
        filename=os.path.join(LOG_DIR, f"{ts}.log"),
        maxBytes=2_000_000,
        backupCount=3,
    )
    fh.setLevel(logging.INFO)
    try:
        from pythonjsonlogger import jsonlogger
        fh.setFormatter(jsonlogger.JsonFormatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s"
        ))
    except Exception:
        fh.setFormatter(logging.Formatter(
            "[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s"
        ))
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fh.formatter)
    sh.setLevel(logging.INFO)
    root.addHandler(fh)
    root.addHandler(sh)
    root.setLevel(logging.INFO)
    root._airbnb_configured = True
configure()
