import os

from dotenv import load_dotenv
from google import genai
import json
import logging
from pathlib import Path
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from logging_config import configure_logging
from prompt import ROUTER_PROMPT, SQL_GENERATION_PROMPT

ROOT_DIR = Path(__file__).resolve().parents[1]
SEMANTIC_LAYER_PATH = ROOT_DIR / "data" / "semantic_layer.json"

configure_logging()
logger = logging.getLogger(__name__)


class NLToSQLState(TypedDict, total=False):
    user_question: str
    model_name: str
    semantic_layer: dict[str, Any]
    router_prompt: str
    router_response_text: str
    router_response: dict[str, Any]
    selected_tables: list[str]
    selected_metrics: list[str]
    sql_context: str
    sql_prompt: str
    sql_response_text: str
    sql_response: dict[str, Any]


def gemini_call(model_name, contents):
    load_dotenv(override=True)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Update your .env file or environment variables.")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config={
            "temperature": 0,
            "top_p": 0.1,
            "seed": 42,
        },
    )
    return response.text


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def load_string_as_json(input_string):
    cleaned_string = input_string.strip()

    if cleaned_string.startswith("```json"):
        cleaned_string = cleaned_string.removeprefix("```json").strip()
    if cleaned_string.startswith("```"):
        cleaned_string = cleaned_string.removeprefix("```").strip()
    if cleaned_string.endswith("```"):
        cleaned_string = cleaned_string.removesuffix("```").strip()

    return json.loads(cleaned_string)


def build_router_tables(semantic_layer):
    tables = {}

    for table_name, table_info in semantic_layer["tables"].items():
        tables[table_name] = {
            "description": table_info.get("description", "No description available"),
            "synonyms": table_info.get("synonyms", []),
            "business_context": table_info.get("business_context", "No business context available"),
        }

    return tables


def build_router_metrics(semantic_layer):
    metrics = {}

    for metric_name, metric_info in semantic_layer.get("metrics", {}).items():
        metrics[metric_name] = {
            "description": metric_info.get("description", "No description available"),
            "synonyms": metric_info.get("synonyms", []),
        }

    return metrics


def select_valid_tables(router_response, semantic_layer):
    available_tables = set(semantic_layer["tables"].keys())
    selected_tables = router_response.get("tables", [])

    valid_tables = [table for table in selected_tables if table in available_tables]
    invalid_tables = sorted(set(selected_tables) - available_tables)

    if invalid_tables:
        logger.warning("Router selected tables not present in semantic layer: %s", invalid_tables)

    logger.info("Selected tables for SQL context: %s", valid_tables)
    return valid_tables


def select_valid_metrics(router_response, semantic_layer):
    available_metrics = set(semantic_layer.get("metrics", {}).keys())
    selected_metrics = router_response.get("metrics", [])

    valid_metrics = [metric for metric in selected_metrics if metric in available_metrics]
    invalid_metrics = sorted(set(selected_metrics) - available_metrics)

    if invalid_metrics:
        logger.warning("Router selected metrics not present in semantic layer: %s", invalid_metrics)

    logger.info("Selected metrics for SQL context: %s", valid_metrics)
    return valid_metrics


def filter_join_paths_for_tables(selected_tables, semantic_layer):
    selected_table_set = set(selected_tables)
    filtered_join_paths = {}

    for path_name, path_info in semantic_layer.get("join_paths", {}).items():
        steps = path_info.get("steps", [])
        path_tables = {
            table
            for step in steps
            for table in (step.get("from"), step.get("to"))
            if table
        }

        if path_tables and path_tables.issubset(selected_table_set):
            filtered_join_paths[path_name] = path_info

    return filtered_join_paths


def build_sql_context(selected_tables, selected_metrics, semantic_layer):
    context = []

    table_context = {}
    for table in selected_tables:
        table_context[table] = semantic_layer["tables"][table]

    context.append(f"tables: {json.dumps(table_context, indent=2)}")

    metric_context = {}
    for metric in selected_metrics:
        metric_context[metric] = semantic_layer["metrics"][metric]

    if metric_context:
        context.append(f"metrics: {json.dumps(metric_context, indent=2)}")

    join_path_context = filter_join_paths_for_tables(selected_tables, semantic_layer)
    if join_path_context:
        context.append(f"join_paths: {json.dumps(join_path_context, indent=2)}")

    identity_columns = build_identity_column_context(semantic_layer)
    context.append(f"identity_columns: {json.dumps(identity_columns, indent=2)}")

    for key in ("ambiguity_rules", "query_hints"):
        if key in semantic_layer:
            context.append(f"{key}: {json.dumps(semantic_layer[key], indent=2)}")

    return "\n\n".join(context)


def pick_display_column(table_info):
    columns = table_info.get("columns", {})

    for column_name in ("name", "invoice_number", "po_number", "grn_number", "code", "reference_number"):
        if column_name in columns:
            return column_name

    return None


def singularize_table_name(table_name):
    if table_name.endswith("ies"):
        return f"{table_name[:-3]}y"
    if table_name.endswith("s"):
        return table_name[:-1]
    return table_name


def entity_alias_for_table(table_name):
    return {
        "purchase_orders": "po",
        "invoices": "invoice",
        "grns": "grn",
    }.get(table_name, singularize_table_name(table_name))


def label_alias_for_entity(entity_name, display_column):
    if display_column.endswith("_number") or display_column == "reference_number":
        return display_column
    return f"{entity_name}_{display_column}"


def build_identity_column_context(semantic_layer):
    identity_columns = {}

    for table_name, table_info in semantic_layer["tables"].items():
        primary_key = table_info.get("primary_key")
        display_column = pick_display_column(table_info)

        if not primary_key or "," in primary_key or not display_column:
            continue

        entity_name = entity_alias_for_table(table_name)
        identity_columns[entity_name] = {
            "table": table_name,
            "id_column": primary_key,
            "display_column": display_column,
            "select_as": {
                "id": f"{entity_name}_id",
                "label": label_alias_for_entity(entity_name, display_column),
            },
        }

    return identity_columns


def create_router_prompt(semantic_layer, user_question):
    return (
        ROUTER_PROMPT
        .replace("{{list_of_tables_from_semantic_layer}}", json.dumps(build_router_tables(semantic_layer), indent=2))
        .replace("{{list_of_metrics_from_semantic_layer}}", json.dumps(build_router_metrics(semantic_layer), indent=2))
        .replace("{{user_question}}", user_question)
    )


def create_sql_prompt(context, user_question):
    return (
        SQL_GENERATION_PROMPT
        .replace("{{context}}", context)
        .replace("{{user_question}}", user_question)
    )


def no_valid_tables_response():
    return {
        "SQL": None,
        "Explanation": "No valid database table was selected for this question.",
        "Assumptions": "The router did not map the question to any table present in the semantic layer.",
        "Followup_Questions": (
            "Can you rephrase the question using available business entities like invoices, "
            "payments, vendors, purchase orders, products, departments, companies, GRNs, "
            "or approval matrix?"
        ),
        "Chart": "none",
    }


def load_semantic_layer_node(state: NLToSQLState):
    return {"semantic_layer": load_json(SEMANTIC_LAYER_PATH)}


def route_question_node(state: NLToSQLState):
    semantic_layer = state["semantic_layer"]
    user_question = state["user_question"]
    model_name = state["model_name"]

    router_prompt = create_router_prompt(semantic_layer, user_question)
    router_response_text = gemini_call(model_name, router_prompt)
    router_response = load_string_as_json(router_response_text)

    logger.info("Router response: %s", router_response)

    return {
        "router_prompt": router_prompt,
        "router_response_text": router_response_text,
        "router_response": router_response,
    }


def select_semantic_context_node(state: NLToSQLState):
    semantic_layer = state["semantic_layer"]
    router_response = state["router_response"]

    selected_tables = select_valid_tables(router_response, semantic_layer)
    selected_metrics = select_valid_metrics(router_response, semantic_layer)

    if not selected_tables:
        logger.warning("No valid tables selected, skipping SQL generation")
        return {
            "selected_tables": selected_tables,
            "selected_metrics": selected_metrics,
            "sql_response": no_valid_tables_response(),
        }

    return {
        "selected_tables": selected_tables,
        "selected_metrics": selected_metrics,
        "sql_context": build_sql_context(selected_tables, selected_metrics, semantic_layer),
    }


def should_generate_sql(state: NLToSQLState) -> Literal["generate_sql", "finish"]:
    if state.get("selected_tables"):
        return "generate_sql"
    return "finish"


def generate_sql_node(state: NLToSQLState):
    sql_prompt = create_sql_prompt(state["sql_context"], state["user_question"])
    sql_response_text = gemini_call(state["model_name"], sql_prompt)
    sql_response = load_string_as_json(sql_response_text)

    logger.info("Generated SQL: %s", sql_response.get("SQL"))

    return {
        "sql_prompt": sql_prompt,
        "sql_response_text": sql_response_text,
        "sql_response": sql_response,
    }


def build_nl_to_sql_graph():
    graph = StateGraph(NLToSQLState)
    graph.add_node("load_semantic_layer", load_semantic_layer_node)
    graph.add_node("route_question", route_question_node)
    graph.add_node("select_semantic_context", select_semantic_context_node)
    graph.add_node("generate_sql", generate_sql_node)

    graph.add_edge(START, "load_semantic_layer")
    graph.add_edge("load_semantic_layer", "route_question")
    graph.add_edge("route_question", "select_semantic_context")
    graph.add_conditional_edges(
        "select_semantic_context",
        should_generate_sql,
        {
            "generate_sql": "generate_sql",
            "finish": END,
        },
    )
    graph.add_edge("generate_sql", END)

    return graph.compile()


NL_TO_SQL_GRAPH = build_nl_to_sql_graph()


def generate_sql_for_question(user_question, model_name="gemini-3-flash-preview"):
    result = NL_TO_SQL_GRAPH.invoke(
        {
            "user_question": user_question,
            "model_name": model_name,
        }
    )
    return result["sql_response"]

if __name__ == "__main__":
    user_question = "How many invoices were raised last month?"
    print(json.dumps(generate_sql_for_question(user_question), indent=2))
