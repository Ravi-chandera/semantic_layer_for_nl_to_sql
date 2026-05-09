from dotenv import load_dotenv
from google import genai
import json
import logging
from pathlib import Path

from logging_config import configure_logging
from prompt import ROUTER_PROMPT, SQL_GENERATION_PROMPT

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[1]
SEMANTIC_LAYER_PATH = ROOT_DIR / "data" / "semantic_layer.json"

configure_logging()
logger = logging.getLogger(__name__)

client = genai.Client()


def gemini_call(model_name, contents):
    response = client.models.generate_content(
        model=model_name, contents=contents
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

    for key in ("ambiguity_rules", "query_hints"):
        if key in semantic_layer:
            context.append(f"{key}: {json.dumps(semantic_layer[key], indent=2)}")

    return "\n\n".join(context)


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


def generate_sql_for_question(user_question, model_name="gemini-3-flash-preview"):
    semantic_layer = load_json(SEMANTIC_LAYER_PATH)

    router_prompt = create_router_prompt(semantic_layer, user_question)
    router_response_text = gemini_call(model_name, router_prompt)
    router_response = load_string_as_json(router_response_text)

    logger.info("Router response: %s", router_response)

    selected_tables = select_valid_tables(router_response, semantic_layer)
    selected_metrics = select_valid_metrics(router_response, semantic_layer)

    if not selected_tables:
        logger.warning("No valid tables selected, skipping SQL generation")
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

    sql_context = build_sql_context(selected_tables, selected_metrics, semantic_layer)

    sql_prompt = create_sql_prompt(sql_context, user_question)
    sql_response_text = gemini_call(model_name, sql_prompt)
    sql_response = load_string_as_json(sql_response_text)

    logger.info("Generated SQL: %s", sql_response.get("SQL"))
    return sql_response

if __name__ == "__main__":
    user_question = "How many invoices were raised last month?"
    print(json.dumps(generate_sql_for_question(user_question), indent=2))
