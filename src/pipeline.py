import os
import uuid

from dotenv import load_dotenv
from google import genai
import json
import logging
from functools import lru_cache
from typing import Any, Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from cache_store import delete_cache_entry, lookup_cache, semantic_layer_hash, store_cache_entry
from logging_config import configure_logging
from langfuse_tracing import safe_update_observation, traced_generation, traced_span
from pipeline_config import (
    INSUFFICIENT_DATA_PREFIX,
    MAX_CLARIFICATION_ATTEMPTS,
    MAX_MEMORY_FIELD_CHARS,
    MAX_MEMORY_SQL_CHARS,
    MAX_MEMORY_TURNS,
    RESULT_SAMPLE_ROWS,
    ROOT_DIR,
    SEMANTIC_LAYER_PATH,
)
from pipeline_memory import (
    active_conversation_context,
    clarification_attempts_for_current_question,
    compact_turn_for_memory,
    format_conversation_context,
    latest_pending_clarification_turn,
    sql_generation_conversation_context,
    summarize_sql_result,
    trim_conversation_turns,
    truncate_text,
)
from pipeline_prompt_builders import (
    create_clarification_prompt,
    create_question_resolution_prompt,
    create_router_prompt,
    create_sql_prompt,
)
from pipeline_responses import (
    EXECUTABLE_SQL_PATTERN,
    build_sql_response,
    clarification_limit_response,
    clarification_needed_response,
    is_executable_sql,
    non_executable_sql_response,
    no_valid_tables_response,
    normalize_sql_response_after_generation,
)
from pipeline_semantic_context import (
    build_identity_column_context,
    build_router_metrics,
    build_router_tables,
    build_sql_context,
    entity_alias_for_table,
    filter_join_paths_for_tables,
    find_required_clarification_rule,
    label_alias_for_entity,
    pick_display_column,
    select_valid_metrics,
    select_valid_tables,
    singularize_table_name,
)

configure_logging()
logger = logging.getLogger(__name__)


class NLToSQLState(TypedDict, total=False):
    user_question: str
    resolved_question: str
    model_name: str
    semantic_layer: dict[str, Any]
    semantic_layer_hash: str
    router_tables_json: str
    router_metrics_json: str
    conversation_turns: list[dict[str, Any]]
    conversation_context: str
    question_resolution_prompt: str
    question_resolution_response_text: str
    question_resolution: dict[str, Any]
    router_prompt: str
    router_response_text: str
    router_response: dict[str, Any]
    selected_tables: list[str]
    selected_metrics: list[str]
    sql_context: str
    clarification_prompt: str
    clarification_response_text: str
    clarification_response: dict[str, Any]
    clarification_attempts: int
    clarification_blocks_sql: bool
    sql_prompt: str
    sql_response_text: str
    sql_response: dict[str, Any]
    cache_lookup: dict[str, Any]
    cache_hit: bool
    cache_strategy: str
    cache_score: float
    cache_store: dict[str, Any]


@lru_cache(maxsize=1)
def load_environment():
    load_dotenv(ROOT_DIR / ".env", override=True)


@lru_cache(maxsize=4)
def get_gemini_client(api_key):
    return genai.Client(api_key=api_key)


def gemini_call(model_name, contents, trace_name="gemini-generate-content"):
    load_environment()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Update your .env file or environment variables.")

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
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=model_parameters,
        )
        safe_update_observation(generation, output=response.text)
        return response.text


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


@lru_cache(maxsize=4)
def load_semantic_layer_bundle(file_path, mtime_ns):
    semantic_layer = load_json(file_path)
    layer_hash = semantic_layer_hash(semantic_layer)
    return {
        "semantic_layer": semantic_layer,
        "semantic_layer_hash": layer_hash,
        "router_tables_json": json.dumps(build_router_tables(semantic_layer), indent=2),
        "router_metrics_json": json.dumps(build_router_metrics(semantic_layer), indent=2),
    }


def load_string_as_json(input_string):
    cleaned_string = input_string.strip()

    if cleaned_string.startswith("```json"):
        cleaned_string = cleaned_string.removeprefix("```json").strip()
    if cleaned_string.startswith("```"):
        cleaned_string = cleaned_string.removeprefix("```").strip()
    if cleaned_string.endswith("```"):
        cleaned_string = cleaned_string.removesuffix("```").strip()

    return json.loads(cleaned_string)


def load_semantic_layer_node(state: NLToSQLState):
    with traced_span(
        "load-semantic-layer",
        input={"path": str(SEMANTIC_LAYER_PATH)},
    ) as span:
        semantic_layer_bundle = load_semantic_layer_bundle(
            str(SEMANTIC_LAYER_PATH),
            SEMANTIC_LAYER_PATH.stat().st_mtime_ns,
        )
        semantic_layer = semantic_layer_bundle["semantic_layer"]
        safe_update_observation(
            span,
            output={
                "table_count": len(semantic_layer.get("tables", {})),
                "metric_count": len(semantic_layer.get("metrics", {})),
                "semantic_layer_hash": semantic_layer_bundle["semantic_layer_hash"],
            },
        )
        return semantic_layer_bundle


def prepare_memory_context_node(state: NLToSQLState):
    with traced_span(
        "prepare-conversation-memory",
        input={"memory_turn_count": len(state.get("conversation_turns", []))},
    ) as span:
        conversation_turns = state.get("conversation_turns", [])
        conversation_context = format_conversation_context(conversation_turns)
        safe_update_observation(
            span,
            output={
                "context": conversation_context,
                "memory_turn_count": len(conversation_turns),
            },
        )
        return {"conversation_context": conversation_context}


def resolve_question_node(state: NLToSQLState):
    with traced_span(
        "resolve-follow-up-question",
        input={
            "user_question": state["user_question"],
            "memory_turn_count": len(state.get("conversation_turns", [])),
        },
    ) as span:
        user_question = state["user_question"]
        conversation_turns = state.get("conversation_turns", [])
        conversation_context = state.get("conversation_context", "No prior conversation.")

        if not conversation_turns:
            node_output = {
                "resolved_question": user_question,
                "question_resolution": {
                    "is_follow_up": False,
                    "standalone_question": user_question,
                    "memory_used": None,
                    "clarification_needed": False,
                    "clarifying_question": None,
                },
            }
            safe_update_observation(span, output=node_output)
            return node_output

        question_resolution_prompt = create_question_resolution_prompt(
            conversation_context,
            user_question,
        )

        try:
            response_text = gemini_call(
                state["model_name"],
                question_resolution_prompt,
                trace_name="gemini-question-resolution",
            )
            question_resolution = load_string_as_json(response_text)
        except Exception as e:
            logger.warning("Question resolution failed; using original question: %s", e)
            node_output = {
                "resolved_question": user_question,
                "question_resolution_prompt": question_resolution_prompt,
                "question_resolution": {
                    "is_follow_up": False,
                    "standalone_question": user_question,
                    "memory_used": None,
                    "clarification_needed": False,
                    "clarifying_question": None,
                },
            }
            safe_update_observation(
                span,
                output=node_output,
                level="WARNING",
                status_message=f"Question resolution failed: {e}",
            )
            return node_output

        resolved_question = str(
            question_resolution.get("standalone_question") or user_question
        ).strip()
        clarifying_question = question_resolution.get("clarifying_question")
        if question_resolution.get("clarification_needed") and not clarifying_question:
            clarifying_question = "Can you clarify what you want to analyze?"

        node_output = {
            "resolved_question": resolved_question or user_question,
            "question_resolution_prompt": question_resolution_prompt,
            "question_resolution_response_text": response_text,
            "question_resolution": question_resolution,
        }

        if question_resolution.get("clarification_needed"):
            node_output["sql_response"] = clarification_needed_response(
                clarifying_question,
                clarification_attempts=1,
                reason="The latest message could not be resolved into an answerable analytics question.",
            )
            node_output["selected_tables"] = []
            node_output["selected_metrics"] = []

        safe_update_observation(span, output=node_output)
        return node_output


def should_route_after_resolution(state: NLToSQLState) -> Literal["route_question", "finish"]:
    question_resolution = state.get("question_resolution", {})

    if question_resolution.get("clarification_needed"):
        return "finish"

    return "route_question"


def lookup_cache_node(state: NLToSQLState):
    with traced_span(
        "lookup-nl-to-sql-cache",
        input={
            "original_question": state["user_question"],
            "resolved_question": state.get("resolved_question") or state["user_question"],
        },
    ) as span:
        question_resolution = state.get("question_resolution", {})
        if question_resolution.get("clarification_needed"):
            node_output = {
                "cache_hit": False,
                "cache_lookup": {
                    "skipped": True,
                    "reason": "Clarification is needed before SQL generation.",
                },
            }
            safe_update_observation(span, output=node_output)
            return node_output

        layer_hash = state.get("semantic_layer_hash") or semantic_layer_hash(state["semantic_layer"])
        resolved_question = state.get("resolved_question") or state["user_question"]
        forced_clarification = find_required_clarification_rule(
            state["semantic_layer"],
            state.get("selected_tables", []),
            resolved_question,
        )
        if forced_clarification:
            node_output = {
                "cache_hit": False,
                "cache_lookup": {
                    "skipped": True,
                    "reason": (
                        "Resolved question matches a no-default semantic ambiguity rule; "
                        "clarification must run before cache reuse."
                    ),
                    "matched_rule": forced_clarification.get("name"),
                    "semantic_layer_hash": layer_hash,
                },
            }
            safe_update_observation(span, output=node_output)
            return node_output

        hit = lookup_cache(resolved_question, layer_hash)

        if not hit:
            node_output = {
                "cache_hit": False,
                "cache_lookup": {
                    "skipped": False,
                    "hit": False,
                    "semantic_layer_hash": layer_hash,
                },
            }
            safe_update_observation(span, output=node_output)
            return node_output

        logger.info(
            "NL-to-SQL cache hit via %s for resolved question: %s",
            hit["strategy"],
            resolved_question,
        )
        cached_sql_response = normalize_sql_response_after_generation(
            hit["sql_response"],
            state,
        )

        if hit["sql_response"].get("SQL") and not cached_sql_response.get("SQL"):
            delete_result = delete_cache_entry(resolved_question, layer_hash)
            logger.warning(
                "Ignored cache hit with non-executable SQL for resolved question %s: %s",
                resolved_question,
                delete_result,
            )
            node_output = {
                "cache_hit": False,
                "cache_lookup": {
                    "skipped": False,
                    "hit": False,
                    "semantic_layer_hash": layer_hash,
                    "discarded_cache_id": hit["id"],
                    "discard_reason": "Cached SQL was not executable.",
                },
            }
            safe_update_observation(span, output=node_output, level="WARNING")
            return node_output

        node_output = {
            "cache_hit": True,
            "cache_strategy": hit["strategy"],
            "cache_score": hit["score"],
            "cache_lookup": {
                "skipped": False,
                "hit": True,
                "cache_id": hit["id"],
                "strategy": hit["strategy"],
                "score": hit["score"],
                "matched_question": hit["question_text"],
                "semantic_layer_hash": layer_hash,
            },
            "sql_response": cached_sql_response,
            "selected_tables": hit["selected_tables"],
            "selected_metrics": hit["selected_metrics"],
        }
        safe_update_observation(span, output=node_output)
        return node_output


def should_route_after_cache(state: NLToSQLState) -> Literal["route_question", "finish"]:
    question_resolution = state.get("question_resolution", {})

    if question_resolution.get("clarification_needed"):
        return "finish"

    if state.get("cache_hit"):
        return "finish"

    return "route_question"


def route_question_node(state: NLToSQLState):
    with traced_span(
        "route-question-to-semantic-layer",
        input={
            "original_question": state["user_question"],
            "resolved_question": state.get("resolved_question") or state["user_question"],
        },
    ) as span:
        semantic_layer = state["semantic_layer"]
        user_question = state.get("resolved_question") or state["user_question"]
        model_name = state["model_name"]

        router_prompt = create_router_prompt(
            semantic_layer=semantic_layer,
            user_question=user_question,
            conversation_context=active_conversation_context(state),
            original_user_question=state["user_question"],
            router_tables_json=state.get("router_tables_json"),
            router_metrics_json=state.get("router_metrics_json"),
        )
        router_response_text = gemini_call(
            model_name,
            router_prompt,
            trace_name="gemini-semantic-router",
        )
        router_response = load_string_as_json(router_response_text)

        logger.info("Router response: %s", router_response)

        node_output = {
            "router_prompt": router_prompt,
            "router_response_text": router_response_text,
            "router_response": router_response,
        }
        safe_update_observation(span, output=node_output)
        return node_output


def select_semantic_context_node(state: NLToSQLState):
    with traced_span(
        "select-semantic-context",
        input={"router_response": state["router_response"]},
    ) as span:
        semantic_layer = state["semantic_layer"]
        router_response = state["router_response"]

        selected_tables = select_valid_tables(router_response, semantic_layer)
        selected_metrics = select_valid_metrics(router_response, semantic_layer)

        if not selected_tables:
            logger.warning("No valid tables selected, skipping SQL generation")
            node_output = {
                "selected_tables": selected_tables,
                "selected_metrics": selected_metrics,
                "sql_response": no_valid_tables_response(),
            }
            safe_update_observation(
                span,
                output=node_output,
                level="WARNING",
                status_message="Router did not select any valid tables.",
            )
            return node_output

        node_output = {
            "selected_tables": selected_tables,
            "selected_metrics": selected_metrics,
            "sql_context": build_sql_context(selected_tables, selected_metrics, semantic_layer),
        }
        safe_update_observation(span, output=node_output)
        return node_output


def should_generate_sql(state: NLToSQLState) -> Literal["generate_sql", "finish"]:
    if state.get("selected_tables"):
        return "generate_sql"
    return "finish"


def should_evaluate_clarification(state: NLToSQLState) -> Literal["evaluate_clarification", "finish"]:
    if state.get("selected_tables"):
        return "evaluate_clarification"
    return "finish"


def evaluate_clarification_node(state: NLToSQLState):
    with traced_span(
        "evaluate-clarification-need",
        input={
            "original_question": state["user_question"],
            "resolved_question": state.get("resolved_question") or state["user_question"],
            "selected_tables": state.get("selected_tables", []),
            "selected_metrics": state.get("selected_metrics", []),
        },
    ) as span:
        clarification_attempts = clarification_attempts_for_current_question(state)
        forced_clarification = find_required_clarification_rule(
            state["semantic_layer"],
            state.get("selected_tables", []),
            state.get("resolved_question") or state["user_question"],
        )
        if forced_clarification:
            clarifying_question = (
                forced_clarification.get("clarifying_question")
                or "Can you clarify the business meaning you want analyzed?"
            )
            clarification_response = {
                "clarification_needed": True,
                "clarifying_question": clarifying_question,
                "can_proceed": False,
                "default_assumption": None,
                "reason": forced_clarification.get("reason"),
                "unanswerable": False,
                "matched_rule": forced_clarification.get("name"),
            }
            node_output = {
                "clarification_response": clarification_response,
                "clarification_attempts": clarification_attempts,
                "clarification_blocks_sql": True,
            }

            if clarification_attempts < MAX_CLARIFICATION_ATTEMPTS:
                node_output["sql_response"] = clarification_needed_response(
                    clarifying_question,
                    clarification_attempts=clarification_attempts + 1,
                    reason=forced_clarification.get("reason"),
                )
            else:
                node_output["sql_response"] = clarification_limit_response(
                    forced_clarification.get("reason")
                )

            safe_update_observation(span, output=node_output)
            return node_output

        clarification_prompt = create_clarification_prompt(
            context=state["sql_context"],
            user_question=state.get("resolved_question") or state["user_question"],
            conversation_context=active_conversation_context(state),
            original_user_question=state["user_question"],
            clarification_attempts=clarification_attempts,
        )

        try:
            response_text = gemini_call(
                state["model_name"],
                clarification_prompt,
                trace_name="gemini-clarification-gate",
            )
            clarification_response = load_string_as_json(response_text)
        except Exception as e:
            logger.warning("Clarification gate failed; continuing to SQL generation: %s", e)
            node_output = {
                "clarification_prompt": clarification_prompt,
                "clarification_response": {
                    "clarification_needed": False,
                    "clarifying_question": None,
                    "can_proceed": True,
                    "default_assumption": None,
                    "reason": f"Clarification gate failed: {e}",
                    "unanswerable": False,
                },
                "clarification_attempts": clarification_attempts,
                "clarification_blocks_sql": False,
            }
            safe_update_observation(
                span,
                output=node_output,
                level="WARNING",
                status_message=f"Clarification gate failed: {e}",
            )
            return node_output

        can_proceed = bool(clarification_response.get("can_proceed"))
        unanswerable = bool(clarification_response.get("unanswerable"))
        clarification_needed = bool(clarification_response.get("clarification_needed"))
        clarifying_question = clarification_response.get("clarifying_question")
        default_assumption = clarification_response.get("default_assumption")
        reason = clarification_response.get("reason")

        node_output = {
            "clarification_prompt": clarification_prompt,
            "clarification_response_text": response_text,
            "clarification_response": clarification_response,
            "clarification_attempts": clarification_attempts,
            "clarification_blocks_sql": False,
        }

        if clarification_needed and clarification_attempts < MAX_CLARIFICATION_ATTEMPTS:
            next_attempt = clarification_attempts + 1
            if not clarifying_question:
                clarifying_question = "Can you clarify the business meaning you want analyzed?"

            node_output["sql_response"] = clarification_needed_response(
                clarifying_question,
                clarification_attempts=next_attempt,
                reason=reason,
            )
            node_output["clarification_blocks_sql"] = True
            safe_update_observation(span, output=node_output)
            return node_output

        if clarification_needed and clarification_attempts >= MAX_CLARIFICATION_ATTEMPTS:
            if default_assumption or can_proceed:
                clarification_response["clarification_needed"] = False
                clarification_response["can_proceed"] = True
                clarification_response["default_assumption"] = default_assumption or (
                    "Proceeding with the safest available semantic-layer default."
                )
                safe_update_observation(span, output=node_output)
                return node_output

            node_output["sql_response"] = clarification_limit_response(reason)
            node_output["clarification_blocks_sql"] = True
            safe_update_observation(
                span,
                output=node_output,
                level="WARNING",
                status_message="Clarification limit reached.",
            )
            return node_output

        if unanswerable and not can_proceed:
            node_output["sql_response"] = clarification_limit_response(reason)
            node_output["clarification_blocks_sql"] = True
            safe_update_observation(
                span,
                output=node_output,
                level="WARNING",
                status_message="Question marked unanswerable by clarification gate.",
            )
            return node_output

        safe_update_observation(span, output=node_output)
        return node_output


def should_generate_sql_after_clarification(state: NLToSQLState) -> Literal["generate_sql", "finish"]:
    if state.get("clarification_blocks_sql"):
        return "finish"
    if state.get("selected_tables"):
        return "generate_sql"
    return "finish"


def generate_sql_node(state: NLToSQLState):
    with traced_span(
        "generate-sql",
        input={
            "original_question": state["user_question"],
            "resolved_question": state.get("resolved_question") or state["user_question"],
            "selected_tables": state.get("selected_tables", []),
            "selected_metrics": state.get("selected_metrics", []),
        },
    ) as span:
        sql_prompt = create_sql_prompt(
            context=state["sql_context"],
            user_question=state.get("resolved_question") or state["user_question"],
            conversation_context=sql_generation_conversation_context(state),
            original_user_question=state["user_question"],
        )
        sql_response_text = gemini_call(
            state["model_name"],
            sql_prompt,
            trace_name="gemini-sql-generation",
        )
        sql_response = load_string_as_json(sql_response_text)
        sql_response = normalize_sql_response_after_generation(sql_response, state)

        logger.info("Generated SQL: %s", sql_response.get("SQL"))

        node_output = {
            "sql_prompt": sql_prompt,
            "sql_response_text": sql_response_text,
            "sql_response": sql_response,
        }
        safe_update_observation(span, output=node_output)
        return node_output


def store_cache_node(state: NLToSQLState):
    with traced_span(
        "store-nl-to-sql-cache",
        input={
            "original_question": state["user_question"],
            "resolved_question": state.get("resolved_question") or state["user_question"],
            "selected_tables": state.get("selected_tables", []),
            "selected_metrics": state.get("selected_metrics", []),
        },
    ) as span:
        sql_response = state.get("sql_response", {})

        if state.get("cache_hit"):
            node_output = {
                "cache_store": {
                    "stored": False,
                    "reason": "This turn was served from cache.",
                },
            }
            safe_update_observation(span, output=node_output)
            return node_output

        cache_store_result = store_cache_entry(
            question=state.get("resolved_question") or state["user_question"],
            original_question=state["user_question"],
            layer_hash=state.get("semantic_layer_hash") or semantic_layer_hash(state["semantic_layer"]),
            model_name=state["model_name"],
            sql_response=sql_response,
            selected_tables=state.get("selected_tables", []),
            selected_metrics=state.get("selected_metrics", []),
        )
        node_output = {"cache_store": cache_store_result}
        safe_update_observation(span, output=node_output)
        return node_output


def remember_turn_node(state: NLToSQLState):
    with traced_span("remember-conversation-turn") as span:
        sql_response = state.get("sql_response", {})
        question_resolution = state.get("question_resolution", {})
        previous_turns = state.get("conversation_turns", [])
        requires_clarification = bool(sql_response.get("Requires_Clarification"))
        new_turn = {
            "user_question": state["user_question"],
            "resolved_question": state.get("resolved_question") or state["user_question"],
            "is_follow_up": question_resolution.get("is_follow_up", False),
            "memory_used": question_resolution.get("memory_used"),
            "selected_tables": state.get("selected_tables", []),
            "selected_metrics": state.get("selected_metrics", []),
            "sql": sql_response.get("SQL"),
            "explanation": sql_response.get("Explanation"),
            "assumptions": sql_response.get("Assumptions"),
            "followup_questions": sql_response.get("Followup_Questions"),
            "requires_clarification": requires_clarification,
            "clarification_question": sql_response.get("Clarification_Question"),
            "clarification_attempts": sql_response.get("Clarification_Attempts", 0),
            "clarification_limit_reached": sql_response.get("Clarification_Limit_Reached", False),
            "chart": sql_response.get("Chart"),
            "cache_hit": state.get("cache_hit", False),
            "cache_strategy": state.get("cache_strategy"),
        }
        conversation_turns = trim_conversation_turns([*previous_turns, new_turn])
        safe_update_observation(
            span,
            output={
                "latest_turn": compact_turn_for_memory(new_turn),
                "memory_turn_count": len(conversation_turns),
            },
        )
        return {"conversation_turns": conversation_turns}


def build_nl_to_sql_graph(checkpointer=None):
    graph = StateGraph(NLToSQLState)
    graph.add_node("load_semantic_layer", load_semantic_layer_node)
    graph.add_node("prepare_memory_context", prepare_memory_context_node)
    graph.add_node("resolve_question", resolve_question_node)
    graph.add_node("lookup_cache", lookup_cache_node)
    graph.add_node("route_question", route_question_node)
    graph.add_node("select_semantic_context", select_semantic_context_node)
    graph.add_node("evaluate_clarification", evaluate_clarification_node)
    graph.add_node("generate_sql", generate_sql_node)
    graph.add_node("store_cache", store_cache_node)
    graph.add_node("remember_turn", remember_turn_node)

    graph.add_edge(START, "load_semantic_layer")
    graph.add_edge("load_semantic_layer", "prepare_memory_context")
    graph.add_edge("prepare_memory_context", "resolve_question")
    graph.add_edge("resolve_question", "lookup_cache")
    graph.add_conditional_edges(
        "lookup_cache",
        should_route_after_cache,
        {
            "route_question": "route_question",
            "finish": "remember_turn",
        },
    )
    graph.add_edge("route_question", "select_semantic_context")
    graph.add_conditional_edges(
        "select_semantic_context",
        should_evaluate_clarification,
        {
            "evaluate_clarification": "evaluate_clarification",
            "finish": "remember_turn",
        },
    )
    graph.add_conditional_edges(
        "evaluate_clarification",
        should_generate_sql_after_clarification,
        {
            "generate_sql": "generate_sql",
            "finish": "remember_turn",
        },
    )
    graph.add_edge("generate_sql", "store_cache")
    graph.add_edge("store_cache", "remember_turn")
    graph.add_edge("remember_turn", END)

    return graph.compile(checkpointer=checkpointer)


NL_TO_SQL_MEMORY = InMemorySaver()
NL_TO_SQL_GRAPH = build_nl_to_sql_graph(checkpointer=NL_TO_SQL_MEMORY)


def build_thread_config(thread_id):
    return {"configurable": {"thread_id": thread_id}}


def generate_sql_for_question(user_question, model_name="gemini-3-flash-preview", thread_id=None):
    resolved_thread_id = thread_id or f"single-turn-{uuid.uuid4()}"
    result = NL_TO_SQL_GRAPH.invoke(
        {
            "user_question": user_question,
            "model_name": model_name,
        },
        build_thread_config(resolved_thread_id),
    )
    sql_response = dict(result["sql_response"])
    question_resolution = result.get("question_resolution", {})

    sql_response["Original_Question"] = result.get("user_question")
    sql_response["Resolved_Question"] = result.get("resolved_question")
    sql_response["Is_Followup"] = question_resolution.get("is_follow_up", False)
    sql_response["Memory_Used"] = question_resolution.get("memory_used")
    sql_response["Selected_Tables"] = result.get("selected_tables", [])
    sql_response["Selected_Metrics"] = result.get("selected_metrics", [])
    sql_response["Cache_Hit"] = result.get("cache_hit", False)
    sql_response["Cache_Strategy"] = result.get("cache_strategy")
    sql_response["Cache_Score"] = result.get("cache_score")

    clarification_response = result.get("clarification_response", {})
    sql_response["Clarification_Decision"] = clarification_response or None
    sql_response["Requires_Clarification"] = bool(sql_response.get("Requires_Clarification"))
    sql_response["Clarification_Question"] = sql_response.get("Clarification_Question")
    sql_response["Clarification_Attempts"] = sql_response.get("Clarification_Attempts", 0)
    sql_response["Clarification_Limit_Reached"] = bool(
        sql_response.get("Clarification_Limit_Reached", False)
    )

    return sql_response


def record_sql_execution_for_thread(thread_id, sql_result):
    if not thread_id:
        return

    config = build_thread_config(thread_id)
    snapshot = NL_TO_SQL_GRAPH.get_state(config)
    values = snapshot.values or {}
    conversation_turns = list(values.get("conversation_turns", []))

    if not conversation_turns:
        return

    latest_turn = dict(conversation_turns[-1])
    latest_turn["result_summary"] = summarize_sql_result(sql_result)
    conversation_turns[-1] = latest_turn

    if isinstance(sql_result, str) and latest_turn.get("sql"):
        semantic_layer = values.get("semantic_layer")
        if semantic_layer:
            delete_result = delete_cache_entry(
                latest_turn.get("resolved_question") or latest_turn.get("user_question"),
                values.get("semantic_layer_hash") or semantic_layer_hash(semantic_layer),
            )
            logger.info("Removed failed SQL from cache: %s", delete_result)

    NL_TO_SQL_GRAPH.update_state(
        config,
        {"conversation_turns": trim_conversation_turns(conversation_turns)},
        as_node="remember_turn",
    )


def get_conversation_memory(thread_id):
    if not thread_id:
        return []

    snapshot = NL_TO_SQL_GRAPH.get_state(build_thread_config(thread_id))
    values = snapshot.values or {}
    return values.get("conversation_turns", [])


def clear_conversation_memory(thread_id):
    if thread_id:
        NL_TO_SQL_MEMORY.delete_thread(thread_id)


def restore_conversation_memory(thread_id, conversation_turns):
    if not thread_id:
        return

    NL_TO_SQL_GRAPH.update_state(
        build_thread_config(thread_id),
        {"conversation_turns": trim_conversation_turns(conversation_turns or [])},
        as_node="remember_turn",
    )

if __name__ == "__main__":
    user_question = "How many invoices were raised last month?"
    print(json.dumps(generate_sql_for_question(user_question), indent=2))
