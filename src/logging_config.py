import logging
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
LOG_FILE_PATH = ROOT_DIR / "logs" / "app.log"
LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


def configure_logging():
    LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_FILE_PATH, mode="a", encoding="utf-8"),
        ],
    )
