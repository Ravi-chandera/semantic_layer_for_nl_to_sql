import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_GEMINI_MODEL = "gemini-3-flash-preview"
MODEL_ENV_VAR = "SEMANTIC_LAYER_MODEL_NAME"
LEGACY_MODEL_ENV_VAR = "GEMINI_MODEL_NAME"
API_KEY_ENV_VAR = "GEMINI_API_KEY"
DEMO_MODE_ENV_VAR = "SEMANTIC_LAYER_DEMO_MODE"


@lru_cache(maxsize=1)
def load_environment():
    load_dotenv(ROOT_DIR / ".env", override=True)


def get_default_model_name():
    load_environment()
    return (
        os.getenv(MODEL_ENV_VAR)
        or os.getenv(LEGACY_MODEL_ENV_VAR)
        or DEFAULT_GEMINI_MODEL
    )


def get_gemini_api_key():
    load_environment()
    return os.getenv(API_KEY_ENV_VAR)


def is_demo_mode_enabled():
    load_environment()
    return os.getenv(DEMO_MODE_ENV_VAR, "").strip().lower() in {"1", "true", "yes", "on"}


def require_gemini_api_key():
    api_key = get_gemini_api_key()
    if api_key:
        return api_key

    demo_hint = (
        f" {DEMO_MODE_ENV_VAR}=true is set; live Gemini calls are disabled in demo mode."
        if is_demo_mode_enabled()
        else ""
    )
    raise RuntimeError(
        f"{API_KEY_ENV_VAR} is not set. Update your .env file or environment variables."
        f"{demo_hint}"
    )

