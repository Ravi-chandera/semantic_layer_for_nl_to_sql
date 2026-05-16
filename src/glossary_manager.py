import json
from copy import deepcopy
from pathlib import Path

try:
    from .pipeline_config import SEMANTIC_LAYER_PATH
except ImportError:
    from pipeline_config import SEMANTIC_LAYER_PATH


METRIC_LIST_FIELDS = ("synonyms", "tables", "examples")
RULE_LIST_FIELDS = ("trigger_phrases", "resolved_by_phrases", "applies_to_tables")


def split_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        raw_items = value
    else:
        raw_items = str(value).split(",")
    return [str(item).strip() for item in raw_items if str(item).strip()]


def join_list(value):
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return value or ""


def load_semantic_layer(path=SEMANTIC_LAYER_PATH):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))


def save_semantic_layer(semantic_layer, path=SEMANTIC_LAYER_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(semantic_layer, indent=2, default=str), encoding="utf-8")
    return path


def metric_records_from_layer(semantic_layer):
    records = []
    for metric_name, metric in semantic_layer.get("metrics", {}).items():
        records.append(
            {
                "metric_name": metric_name,
                "description": metric.get("description") or "",
                "formula": metric.get("formula") or metric.get("sql") or "",
                "filters": metric.get("filters") or "",
                "unit": metric.get("unit") or metric.get("result_unit") or "",
                "owner": metric.get("owner") or "",
                "examples": join_list(metric.get("examples")),
                "ambiguity_rules": metric.get("ambiguity_rules") or "",
                "synonyms": join_list(metric.get("synonyms")),
                "tables": join_list(metric.get("tables")),
                "enabled": metric.get("enabled", True),
            }
        )
    return records


def ambiguity_rule_records_from_layer(semantic_layer):
    records = []
    for rule_name, rule in semantic_layer.get("ambiguity_rules", {}).items():
        dimensions = rule.get("ambiguous_dimensions") or []
        records.append(
            {
                "rule_name": rule_name,
                "trigger_phrases": join_list(rule.get("trigger_phrases")),
                "resolved_by_phrases": join_list(rule.get("resolved_by_phrases")),
                "applies_to_tables": join_list(rule.get("applies_to_tables")),
                "ambiguous_dimensions_json": json.dumps(dimensions, indent=2),
                "clarification_question": rule.get("clarification_question") or "",
                "default_assumption": rule.get("default_assumption") or "",
                "reason": rule.get("reason") or "",
                "enabled": rule.get("enabled", True),
            }
        )
    return records


def _editor_records(value):
    if isinstance(value, list):
        return value
    if hasattr(value, "to_dict"):
        return value.to_dict("records")
    if isinstance(value, dict) and "data" in value:
        return value["data"]
    return []


def _clean_text(value):
    if value is None:
        return ""
    return str(value).strip()


def _parse_ambiguous_dimensions(value):
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return value

    parsed = json.loads(str(value))
    if not isinstance(parsed, list):
        raise ValueError("ambiguous_dimensions_json must be a JSON list")
    return parsed


def normalize_metric_record(row, existing_metric=None):
    existing_metric = deepcopy(existing_metric or {})
    formula = _clean_text(row.get("formula") or row.get("sql") or existing_metric.get("sql"))
    filters = _clean_text(row.get("filters"))
    unit = _clean_text(row.get("unit") or row.get("result_unit") or existing_metric.get("result_unit"))

    metric = existing_metric
    metric["description"] = _clean_text(row.get("description") or existing_metric.get("description"))
    metric["sql"] = formula
    metric["formula"] = formula
    metric["filters"] = filters or None
    metric["synonyms"] = split_list(row.get("synonyms"))
    metric["result_unit"] = unit or "number"
    metric["unit"] = unit or metric["result_unit"]
    metric["tables"] = split_list(row.get("tables"))
    metric["owner"] = _clean_text(row.get("owner"))
    metric["examples"] = split_list(row.get("examples"))
    metric["ambiguity_rules"] = _clean_text(row.get("ambiguity_rules"))
    metric["enabled"] = bool(row.get("enabled", True))
    return metric


def normalize_ambiguity_rule_record(row, existing_rule=None):
    existing_rule = deepcopy(existing_rule or {})
    default_assumption = _clean_text(row.get("default_assumption"))

    rule = existing_rule
    for field in RULE_LIST_FIELDS:
        rule[field] = split_list(row.get(field))
    rule["ambiguous_dimensions"] = _parse_ambiguous_dimensions(
        row.get("ambiguous_dimensions_json", existing_rule.get("ambiguous_dimensions", []))
    )
    rule["clarification_question"] = _clean_text(row.get("clarification_question"))
    rule["default_assumption"] = default_assumption or None
    rule["reason"] = _clean_text(row.get("reason"))
    rule["enabled"] = bool(row.get("enabled", True))
    return rule


def validate_metric_records(metric_rows, available_tables=None):
    available_tables = set(available_tables or [])
    seen = set()
    errors = []
    warnings = []

    for index, row in enumerate(_editor_records(metric_rows), start=1):
        metric_name = _clean_text(row.get("metric_name"))
        if not metric_name:
            errors.append(f"Metric row {index} is missing metric_name.")
            continue

        if metric_name in seen:
            errors.append(f"Metric '{metric_name}' is duplicated.")
        seen.add(metric_name)

        formula = _clean_text(row.get("formula") or row.get("sql"))
        if not formula:
            errors.append(f"Metric '{metric_name}' is missing a formula.")

        unknown_tables = sorted(set(split_list(row.get("tables"))) - available_tables)
        if available_tables and unknown_tables:
            warnings.append(
                f"Metric '{metric_name}' references unknown tables: {', '.join(unknown_tables)}."
            )

    return {"errors": errors, "warnings": warnings}


def apply_glossary_records(semantic_layer, metric_rows, rule_rows):
    updated = deepcopy(semantic_layer)
    previous_metrics = updated.get("metrics", {})
    previous_rules = updated.get("ambiguity_rules", {})

    metrics = {}
    for row in _editor_records(metric_rows):
        metric_name = _clean_text(row.get("metric_name"))
        if not metric_name:
            continue
        metric = normalize_metric_record(row, previous_metrics.get(metric_name))
        if metric.pop("enabled", True):
            metrics[metric_name] = metric

    rules = {}
    for row in _editor_records(rule_rows):
        rule_name = _clean_text(row.get("rule_name"))
        if not rule_name:
            continue
        rule = normalize_ambiguity_rule_record(row, previous_rules.get(rule_name))
        if rule.pop("enabled", True):
            rules[rule_name] = rule

    updated["metrics"] = metrics
    updated["ambiguity_rules"] = rules
    return updated
