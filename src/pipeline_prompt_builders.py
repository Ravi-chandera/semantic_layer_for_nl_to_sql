import json

try:
    from .pipeline_config import MAX_CLARIFICATION_ATTEMPTS
    from .pipeline_semantic_context import build_router_metrics, build_router_tables
    from .prompt import (
        CLARIFICATION_PROMPT,
        QUESTION_RESOLUTION_PROMPT,
        ROUTER_PROMPT,
        SQL_GENERATION_PROMPT,
    )
except ImportError:
    from pipeline_config import MAX_CLARIFICATION_ATTEMPTS
    from pipeline_semantic_context import build_router_metrics, build_router_tables
    from prompt import (
        CLARIFICATION_PROMPT,
        QUESTION_RESOLUTION_PROMPT,
        ROUTER_PROMPT,
        SQL_GENERATION_PROMPT,
    )


def create_question_resolution_prompt(conversation_context, user_question):
    return (
        QUESTION_RESOLUTION_PROMPT
        .replace("{{conversation_context}}", conversation_context)
        .replace("{{user_question}}", user_question)
    )


def create_router_prompt(
    semantic_layer,
    user_question,
    conversation_context,
    original_user_question,
    router_tables_json=None,
    router_metrics_json=None,
):
    tables_json = router_tables_json or json.dumps(build_router_tables(semantic_layer), indent=2)
    metrics_json = router_metrics_json or json.dumps(build_router_metrics(semantic_layer), indent=2)

    return (
        ROUTER_PROMPT
        .replace("{{list_of_tables_from_semantic_layer}}", tables_json)
        .replace("{{list_of_metrics_from_semantic_layer}}", metrics_json)
        .replace("{{conversation_context}}", conversation_context)
        .replace("{{original_user_question}}", original_user_question)
        .replace("{{user_question}}", user_question)
    )


def create_sql_prompt(context, user_question, conversation_context, original_user_question):
    return (
        SQL_GENERATION_PROMPT
        .replace("{{context}}", context)
        .replace("{{conversation_context}}", conversation_context)
        .replace("{{original_user_question}}", original_user_question)
        .replace("{{user_question}}", user_question)
    )


def create_clarification_prompt(
    context,
    user_question,
    conversation_context,
    original_user_question,
    clarification_attempts,
):
    return (
        CLARIFICATION_PROMPT
        .replace("{{context}}", context)
        .replace("{{conversation_context}}", conversation_context)
        .replace("{{original_user_question}}", original_user_question)
        .replace("{{user_question}}", user_question)
        .replace("{{clarification_attempts}}", str(clarification_attempts))
        .replace("{{max_clarification_attempts}}", str(MAX_CLARIFICATION_ATTEMPTS))
    )
