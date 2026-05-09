import json
import logging


logger = logging.getLogger(__name__)


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
