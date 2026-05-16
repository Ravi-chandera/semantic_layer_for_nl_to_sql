import json
from copy import deepcopy
from datetime import date
from pathlib import Path
from typing import Any

try:
    from .pipeline_config import ROOT_DIR
except ImportError:
    from pipeline_config import ROOT_DIR


SETTINGS_PATH = ROOT_DIR / "data" / "data_settings.json"

DEFAULT_DATA_SETTINGS = {
    "default_currency": "INR",
    "fiscal_year_start": {"month": 4, "day": 1},
    "timezone": "Asia/Kolkata",
    "today_anchor": "2026-05-16",
    "month_definition": "calendar",
    "quarter_definition": "fiscal",
    "display_format": {
        "currency": "code",
        "date": "%Y-%m-%d",
        "month": "%Y-%m",
        "decimal_places": 2,
    },
}

_CURRENCY_SYMBOLS = {
    "INR": "Rs.",
    "USD": "$",
    "EUR": "EUR",
    "GBP": "GBP",
    "JPY": "JPY",
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _int_in_range(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def _valid_date_string(value: Any, default: str) -> str:
    try:
        date.fromisoformat(str(value))
        return str(value)
    except (TypeError, ValueError):
        return default


def normalize_data_settings(settings: dict[str, Any] | None) -> dict[str, Any]:
    normalized = _deep_merge(DEFAULT_DATA_SETTINGS, settings or {})
    defaults = DEFAULT_DATA_SETTINGS

    normalized["default_currency"] = str(
        normalized.get("default_currency") or defaults["default_currency"]
    ).upper()
    normalized["timezone"] = str(normalized.get("timezone") or defaults["timezone"])
    normalized["today_anchor"] = _valid_date_string(
        normalized.get("today_anchor"),
        defaults["today_anchor"],
    )

    fiscal_year_start = normalized.get("fiscal_year_start") or {}
    normalized["fiscal_year_start"] = {
        "month": _int_in_range(
            fiscal_year_start.get("month"),
            default=defaults["fiscal_year_start"]["month"],
            minimum=1,
            maximum=12,
        ),
        "day": _int_in_range(
            fiscal_year_start.get("day"),
            default=defaults["fiscal_year_start"]["day"],
            minimum=1,
            maximum=31,
        ),
    }

    if normalized.get("month_definition") not in {"calendar"}:
        normalized["month_definition"] = defaults["month_definition"]
    if normalized.get("quarter_definition") not in {"calendar", "fiscal"}:
        normalized["quarter_definition"] = defaults["quarter_definition"]

    display_format = normalized.get("display_format") or {}
    normalized["display_format"] = {
        "currency": display_format.get("currency")
        if display_format.get("currency") in {"code", "symbol", "code_suffix"}
        else defaults["display_format"]["currency"],
        "date": str(display_format.get("date") or defaults["display_format"]["date"]),
        "month": str(display_format.get("month") or defaults["display_format"]["month"]),
        "decimal_places": _int_in_range(
            display_format.get("decimal_places"),
            default=defaults["display_format"]["decimal_places"],
            minimum=0,
            maximum=6,
        ),
    }
    return normalized


def load_data_settings(path: Path | str = SETTINGS_PATH) -> dict[str, Any]:
    settings_path = Path(path)
    if not settings_path.exists():
        return normalize_data_settings({})

    with settings_path.open("r", encoding="utf-8") as f:
        return normalize_data_settings(json.load(f))


def save_data_settings(settings: dict[str, Any], path: Path | str = SETTINGS_PATH) -> dict[str, Any]:
    settings_path = Path(path)
    normalized = normalize_data_settings(settings)
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    with settings_path.open("w", encoding="utf-8") as f:
        json.dump(normalized, f, indent=2)
        f.write("\n")
    return normalized


def data_settings_mtime_ns(path: Path | str = SETTINGS_PATH) -> int:
    settings_path = Path(path)
    return settings_path.stat().st_mtime_ns if settings_path.exists() else 0


def format_currency(value: Any, settings: dict[str, Any] | None = None) -> str:
    normalized = normalize_data_settings(settings)
    currency = normalized["default_currency"]
    decimal_places = normalized["display_format"]["decimal_places"]
    if value is None:
        amount_text = "n/a"
    else:
        try:
            amount_text = f"{float(value):,.{decimal_places}f}"
        except (TypeError, ValueError):
            amount_text = str(value)

    style = normalized["display_format"]["currency"]
    if style == "symbol":
        return f"{_CURRENCY_SYMBOLS.get(currency, currency)} {amount_text}"
    if style == "code_suffix":
        return f"{amount_text} {currency}"
    return f"{currency} {amount_text}"


def format_date(value: Any, settings: dict[str, Any] | None = None, *, kind: str = "date") -> str:
    if value in (None, ""):
        return "n/a"

    normalized = normalize_data_settings(settings)
    format_key = "month" if kind == "month" else "date"
    date_format = normalized["display_format"][format_key]
    text = str(value)
    try:
        parsed = date.fromisoformat(text[:10])
    except ValueError:
        return text
    return parsed.strftime(date_format)


def fiscal_year_label(value: Any, settings: dict[str, Any] | None = None) -> str:
    normalized = normalize_data_settings(settings)
    try:
        parsed = date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return "n/a"

    start = normalized["fiscal_year_start"]
    fiscal_start = date(parsed.year, start["month"], min(start["day"], 28))
    start_year = parsed.year if parsed >= fiscal_start else parsed.year - 1
    return f"FY{start_year}-{str(start_year + 1)[-2:]}"


def build_settings_context(settings: dict[str, Any] | None = None) -> dict[str, Any]:
    normalized = normalize_data_settings(settings)
    fiscal_start = normalized["fiscal_year_start"]
    return {
        "default_currency": normalized["default_currency"],
        "timezone": normalized["timezone"],
        "today_anchor": normalized["today_anchor"],
        "relative_date_rule": (
            "Resolve relative periods such as today, yesterday, last 7 days, "
            "last month, this quarter, and YTD from today_anchor in the configured timezone."
        ),
        "fiscal_year_start": f"{fiscal_start['month']:02d}-{fiscal_start['day']:02d}",
        "month_definition": normalized["month_definition"],
        "quarter_definition": normalized["quarter_definition"],
        "display_format": normalized["display_format"],
    }


def settings_hash_payload(settings: dict[str, Any] | None = None) -> dict[str, Any]:
    normalized = normalize_data_settings(settings)
    return {
        "data_settings": normalized,
        "context": build_settings_context(normalized),
    }
