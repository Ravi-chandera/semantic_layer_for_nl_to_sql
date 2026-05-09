import hashlib
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
LANGFUSE_PARENT_SPAN_ID = "0000000000000001"

logger = logging.getLogger(__name__)
_LANGFUSE_CLIENT = None
_LANGFUSE_AVAILABLE = None


def _load_env():
    load_dotenv(ROOT_DIR / ".env", override=True)


def get_langfuse_client():
    global _LANGFUSE_AVAILABLE, _LANGFUSE_CLIENT

    if _LANGFUSE_AVAILABLE is False:
        return None

    if _LANGFUSE_CLIENT is not None:
        return _LANGFUSE_CLIENT

    _load_env()

    try:
        from langfuse import get_client

        _LANGFUSE_CLIENT = get_client()
        _LANGFUSE_AVAILABLE = True
        return _LANGFUSE_CLIENT
    except Exception as e:
        _LANGFUSE_AVAILABLE = False
        logger.warning("Langfuse tracing is disabled: %s", e)
        return None


def create_conversation_trace_id(thread_id: str):
    seed = f"nl-to-sql-conversation:{thread_id}"
    client = get_langfuse_client()

    if client is not None:
        try:
            return client.create_trace_id(seed=seed)
        except Exception as e:
            logger.warning("Failed to create Langfuse trace id with SDK: %s", e)

    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:32]


def _shorten(value: Any, max_chars=180):
    text = "" if value is None else str(value)
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()}..."


def safe_update_observation(observation, **kwargs):
    if observation is None:
        return

    try:
        observation.update(**kwargs)
    except Exception as e:
        logger.warning("Failed to update Langfuse observation: %s", e)


@contextmanager
def conversation_turn_trace(thread_id, user_question, chat_name=None, turn_index=None):
    client = get_langfuse_client()

    if client is None:
        yield None
        return

    trace_id = create_conversation_trace_id(thread_id)
    trace_name = f"NL to SQL: {_shorten(chat_name or user_question, 140)}"
    metadata = {
        "component": "streamlit",
        "threadId": str(thread_id),
    }
    if turn_index is not None:
        metadata["turnIndex"] = str(turn_index)

    try:
        from langfuse import propagate_attributes

        observation_manager = client.start_as_current_observation(
            as_type="span",
            name=f"conversation-turn-{turn_index}" if turn_index else "conversation-turn",
            trace_context={
                "trace_id": trace_id,
                "parent_span_id": LANGFUSE_PARENT_SPAN_ID,
            },
            input={
                "chat_name": chat_name or user_question,
                "thread_id": thread_id,
                "turn_index": turn_index,
                "user_question": user_question,
            },
            metadata=metadata,
        )
        attribute_manager = propagate_attributes(
            session_id=thread_id,
            trace_name=trace_name,
            tags=["nl-to-sql", "streamlit"],
            metadata=metadata,
        )
    except Exception as e:
        logger.warning("Langfuse turn trace failed to start; running without trace: %s", e)
        yield None
        return

    yielded = False
    try:
        with observation_manager as span:
            with attribute_manager:
                yielded = True
                yield span
    except Exception as e:
        if yielded:
            raise
        logger.warning("Langfuse turn trace failed to start; running without trace: %s", e)
        yield None


@contextmanager
def traced_span(name, input=None, metadata=None, as_type="span"):
    client = get_langfuse_client()

    if client is None:
        yield None
        return

    try:
        observation_manager = client.start_as_current_observation(
            as_type=as_type,
            name=name,
            input=input,
            metadata=metadata,
        )
    except Exception as e:
        logger.warning("Langfuse span failed to start; running without span %s: %s", name, e)
        yield None
        return

    yielded = False
    try:
        with observation_manager as span:
            yielded = True
            yield span
    except Exception:
        if yielded:
            raise
        logger.warning("Langfuse span failed to enter; running without span %s", name)
        yield None


@contextmanager
def traced_generation(name, model_name, input=None, metadata=None, model_parameters=None):
    client = get_langfuse_client()

    if client is None:
        yield None
        return

    try:
        observation_manager = client.start_as_current_observation(
            as_type="generation",
            name=name,
            model=model_name,
            input=input,
            metadata=metadata,
            model_parameters=model_parameters,
        )
    except Exception as e:
        logger.warning("Langfuse generation failed to start; running without generation %s: %s", name, e)
        yield None
        return

    yielded = False
    try:
        with observation_manager as generation:
            yielded = True
            yield generation
    except Exception:
        if yielded:
            raise
        logger.warning("Langfuse generation failed to enter; running without generation %s", name)
        yield None


def flush_langfuse():
    client = get_langfuse_client()

    if client is None:
        return

    try:
        client.flush()
    except Exception as e:
        logger.warning("Failed to flush Langfuse traces: %s", e)
