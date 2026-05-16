import json
import logging
import re
from collections import deque

try:
    from .data_settings import build_settings_context, data_settings_mtime_ns, load_data_settings
    from .feedback_store import build_semantic_correction_context
except ImportError:
    from data_settings import build_settings_context, data_settings_mtime_ns, load_data_settings
    from feedback_store import build_semantic_correction_context


logger = logging.getLogger(__name__)
_SQL_CONTEXT_CACHE = {}


def clear_sql_context_cache():
    _SQL_CONTEXT_CACHE.clear()


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
            "tables": metric_info.get("tables", []),
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


def phrase_matches(text, phrase):
    normalized_phrase = str(phrase or "").strip().lower()
    if not normalized_phrase:
        return False

    pattern = rf"(?<!\w){re.escape(normalized_phrase)}(?!\w)"
    return bool(re.search(pattern, text))


def find_required_clarification_rule(semantic_layer, selected_tables, question):
    normalized_question = str(question or "").lower()
    selected_table_set = set(selected_tables or [])

    for rule_name, rule in semantic_layer.get("ambiguity_rules", {}).items():
        if rule.get("default_assumption"):
            continue

        applies_to_tables = set(rule.get("applies_to_tables") or [])
        if applies_to_tables and selected_table_set and selected_table_set.isdisjoint(applies_to_tables):
            continue

        trigger_phrases = rule.get("trigger_phrases") or []
        if any(phrase_matches(normalized_question, phrase) for phrase in trigger_phrases):
            resolved_by_phrases = rule.get("resolved_by_phrases") or []
            if any(phrase_matches(normalized_question, phrase) for phrase in resolved_by_phrases):
                continue

            return {
                "name": rule_name,
                "rule": rule,
                "clarifying_question": rule.get("clarification_question"),
                "options": clarification_options_from_rule(rule_name, rule),
                "reason": rule.get("reason") or f"Matched semantic ambiguity rule: {rule_name}.",
            }

    return None


def _humanize_dimension_label(label):
    return str(label or "").replace("_", " ").strip()


def _title_label(label):
    return _humanize_dimension_label(label).title()


def _default_option_label(rule_name, label):
    lower_name = str(rule_name or "").lower()
    human_label = _humanize_dimension_label(label)

    if "top" in lower_name or "rank" in lower_name:
        return f"Rank by {human_label}"

    return f"Use {_title_label(label)}"


def _default_resolution_text(rule_name, label):
    lower_name = str(rule_name or "").lower()
    human_label = _humanize_dimension_label(label)

    if "top" in lower_name or "rank" in lower_name:
        return f"Rank by {human_label}."

    return f"Use {human_label}."


def clarification_options_from_rule(rule_name, rule):
    options = []

    for index, dimension in enumerate(rule.get("ambiguous_dimensions") or []):
        if not isinstance(dimension, dict):
            continue

        label = dimension.get("label")
        if not label:
            continue

        resolution_text = (
            dimension.get("resolution_text")
            or dimension.get("resolved_text")
            or _default_resolution_text(rule_name, label)
        )
        option = {
            "id": dimension.get("id") or str(label),
            "label": dimension.get("display_label") or _default_option_label(rule_name, label),
            "resolution_text": resolution_text,
            "detail": dimension.get("detail") or dimension.get("description") or dimension.get("sql_hint"),
            "sql_hint": dimension.get("sql_hint"),
            "is_default": bool(
                dimension.get("is_default")
                or rule.get("default_assumption") == label
                or (rule.get("default_assumption") and rule.get("default_assumption") == dimension.get("id"))
            ),
            "position": index,
        }
        options.append(option)

    return options


def metric_source_tables(selected_metrics, semantic_layer):
    source_tables = []
    available_tables = set(semantic_layer["tables"].keys())

    for metric in selected_metrics:
        metric_info = semantic_layer.get("metrics", {}).get(metric, {})
        for table in metric_info.get("tables") or []:
            if table in available_tables and table not in source_tables:
                source_tables.append(table)

    return source_tables


def build_relationship_graph(semantic_layer):
    graph = {table: [] for table in semantic_layer["tables"]}

    def add_edge(left, right, join_condition):
        if left not in graph or right not in graph:
            return

        graph[left].append(
            {
                "table": right,
                "join_condition": join_condition,
            }
        )
        graph[right].append(
            {
                "table": left,
                "join_condition": join_condition,
            }
        )

    for table_name, table_info in semantic_layer["tables"].items():
        for relationship in table_info.get("relationships") or []:
            add_edge(
                table_name,
                relationship.get("target_table"),
                relationship.get("join_condition"),
            )

    for path_info in semantic_layer.get("join_paths", {}).values():
        for step in path_info.get("steps") or []:
            add_edge(step.get("from"), step.get("to"), step.get("on"))

    return graph


def shortest_table_path(start_table, end_table, graph):
    if start_table == end_table:
        return [start_table]

    queue = deque([(start_table, [start_table])])
    visited = {start_table}

    while queue:
        current_table, path = queue.popleft()
        for edge in graph.get(current_table, []):
            next_table = edge["table"]
            if next_table in visited:
                continue

            next_path = [*path, next_table]
            if next_table == end_table:
                return next_path

            visited.add(next_table)
            queue.append((next_table, next_path))

    return []


def expand_selected_tables_for_context(selected_tables, selected_metrics, semantic_layer):
    expanded_tables = []
    available_tables = set(semantic_layer["tables"].keys())

    for table in [*selected_tables, *metric_source_tables(selected_metrics, semantic_layer)]:
        if table in available_tables and table not in expanded_tables:
            expanded_tables.append(table)

    graph = build_relationship_graph(semantic_layer)
    original_tables = list(expanded_tables)

    for index, left_table in enumerate(original_tables):
        for right_table in original_tables[index + 1:]:
            path = shortest_table_path(left_table, right_table, graph)
            for table in path:
                if table not in expanded_tables:
                    expanded_tables.append(table)

    if expanded_tables != selected_tables:
        logger.info(
            "Expanded selected tables from %s to %s using metrics and relationship paths.",
            selected_tables,
            expanded_tables,
        )

    return expanded_tables


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


def build_sql_context(selected_tables, selected_metrics, semantic_layer, data_settings=None):
    if data_settings is None:
        data_settings = load_data_settings()

    cache_key = (
        id(semantic_layer),
        tuple(selected_tables),
        tuple(selected_metrics),
        data_settings_mtime_ns(),
        json.dumps(data_settings, sort_keys=True),
    )
    if cache_key in _SQL_CONTEXT_CACHE:
        return _SQL_CONTEXT_CACHE[cache_key]

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

    context.append(
        "global_data_settings: "
        + json.dumps(build_settings_context(data_settings), indent=2)
    )

    join_path_context = filter_join_paths_for_tables(selected_tables, semantic_layer)
    if join_path_context:
        context.append(f"join_paths: {json.dumps(join_path_context, indent=2)}")

    identity_columns = build_identity_column_context(semantic_layer)
    context.append(f"identity_columns: {json.dumps(identity_columns, indent=2)}")

    for key in ("ambiguity_rules", "query_hints"):
        if key in semantic_layer:
            context.append(f"{key}: {json.dumps(semantic_layer[key], indent=2)}")

    try:
        correction_context = build_semantic_correction_context(
            selected_tables=selected_tables,
            selected_metrics=selected_metrics,
        )
    except Exception as e:
        logger.warning("Could not load semantic correction context: %s", e)
        correction_context = {}

    if correction_context:
        context.append(f"semantic_corrections: {json.dumps(correction_context, indent=2)}")

    sql_context = "\n\n".join(context)
    _SQL_CONTEXT_CACHE[cache_key] = sql_context
    return sql_context


def pick_display_column(table_info):
    columns = table_info.get("columns", {})

    preferred_names = (
        "name",
        "title",
        "label",
        "code",
        "number",
        "reference_number",
        "reference",
    )
    for column_name in preferred_names:
        if column_name in columns and not columns[column_name].get("is_sensitive"):
            return column_name

    for column_name, column_info in columns.items():
        column_type = str(column_info.get("type") or "").upper()
        if column_info.get("is_sensitive"):
            continue
        if any(part in column_type for part in ("CHAR", "TEXT", "CLOB", "VARCHAR")):
            return column_name

    return None


def singularize_table_name(table_name):
    if table_name.endswith("ies"):
        return f"{table_name[:-3]}y"
    if table_name.endswith("s"):
        return table_name[:-1]
    return table_name


def entity_alias_for_table(table_name):
    return singularize_table_name(table_name)


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
