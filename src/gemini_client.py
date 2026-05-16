import logging
import time
from functools import lru_cache

from google import genai

try:
    from .langfuse_tracing import safe_update_observation, traced_generation
    from .logging_config import configure_logging
    from .model_config import get_default_model_name, require_gemini_api_key
    from .model_config import load_environment as load_model_environment
except ImportError:
    from langfuse_tracing import safe_update_observation, traced_generation
    from logging_config import configure_logging
    from model_config import get_default_model_name, require_gemini_api_key
    from model_config import load_environment as load_model_environment

configure_logging()
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_environment():
    load_model_environment()


@lru_cache(maxsize=4)
def get_gemini_client(api_key):
    return genai.Client(api_key=api_key)


def gemini_call(model_name, contents, trace_name="gemini-generate-content"):
    api_key = require_gemini_api_key()
    model_name = model_name or get_default_model_name()
    client = get_gemini_client(api_key)
    model_parameters = {
        "temperature": 0,
        "top_p": 0.1,
        "seed": 42,
    }

    with traced_generation(
        trace_name,
        model_name,
        input={"prompt": contents},
        model_parameters=model_parameters,
    ) as generation:
        last_error = None
        for attempt in range(1, 4):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=model_parameters,
                )
                safe_update_observation(generation, output=response.text)
                return response.text
            except Exception as e:
                last_error = e
                logger.warning(
                    "Gemini call failed for %s on attempt %s/3: %s",
                    trace_name,
                    attempt,
                    e,
                )
                if attempt < 3:
                    time.sleep(1.5 * attempt)

        safe_update_observation(
            generation,
            output={"error": str(last_error)},
            level="ERROR",
            status_message=str(last_error),
        )
        raise last_error
