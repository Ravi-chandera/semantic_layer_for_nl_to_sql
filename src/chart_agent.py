from google import genai
import json
import logging
from functools import lru_cache
from pathlib import Path

import plotly.express as px

try:
    from .logging_config import configure_logging
    from .langfuse_tracing import safe_update_observation, traced_generation, traced_span
    from .model_config import get_default_model_name, load_environment as load_model_environment, require_gemini_api_key
    from .prompt import CHART_AGENT_PROMPT
except ImportError:
    from logging_config import configure_logging
    from langfuse_tracing import safe_update_observation, traced_generation, traced_span
    from model_config import get_default_model_name, load_environment as load_model_environment, require_gemini_api_key
    from prompt import CHART_AGENT_PROMPT

ROOT_DIR = Path(__file__).resolve().parents[1]
CHARTS_DIR = ROOT_DIR / "charts"

configure_logging()
logger = logging.getLogger(__name__)

SUPPORTED_CHART_FUNCTIONS = {
    "bar_chart",
    "line_chart",
    "pie_chart",
    "scatter_chart",
    "none",
}

CHART_FUNCTION_ALIASES = {
    "bar": "bar_chart",
    "line": "line_chart",
    "pie": "pie_chart",
    "scatter": "scatter_chart",
}


@lru_cache(maxsize=1)
def load_environment():
    load_model_environment()


@lru_cache(maxsize=4)
def get_gemini_client(api_key):
    return genai.Client(api_key=api_key)


def gemini_call(model_name, contents, trace_name="gemini-chart-planner"):
    api_key = require_gemini_api_key()
    model_name = model_name or get_default_model_name()
    client = get_gemini_client(api_key)
    with traced_generation(
        trace_name,
        model_name,
        input={"prompt": contents},
    ) as generation:
        response = client.models.generate_content(
            model=model_name, contents=contents
        )
        safe_update_observation(generation, output=response.text)
        return response.text


def load_string_as_json(input_string):
    cleaned_string = input_string.strip()

    if cleaned_string.startswith("```json"):
        cleaned_string = cleaned_string.removeprefix("```json").strip()
    if cleaned_string.startswith("```"):
        cleaned_string = cleaned_string.removeprefix("```").strip()
    if cleaned_string.endswith("```"):
        cleaned_string = cleaned_string.removesuffix("```").strip()

    return json.loads(cleaned_string)


def create_chart_prompt(user_question, chart_hint, sql_result):
    preview_rows = sql_result[:50]

    return (
        CHART_AGENT_PROMPT
        .replace("{{user_question}}", user_question)
        .replace("{{chart_hint}}", chart_hint or "none")
        .replace("{{sql_result}}", json.dumps(preview_rows, indent=2, default=str))
    )


def is_chart_requested(chart_hint):
    return chart_hint and str(chart_hint).strip().lower() != "none"


def is_chartable_result(sql_result):
    return isinstance(sql_result, list) and len(sql_result) > 0 and isinstance(sql_result[0], dict)


def column_exists(sql_result, column_name):
    return column_name in sql_result[0]


def validate_chart_plan(chart_plan, sql_result):
    function_name = CHART_FUNCTION_ALIASES.get(
        chart_plan.get("function_name"),
        chart_plan.get("function_name"),
    )
    chart_plan["function_name"] = function_name
    arguments = chart_plan.get("arguments", {})

    if function_name not in SUPPORTED_CHART_FUNCTIONS:
        raise ValueError(f"Unsupported chart function: {function_name}")

    if function_name == "none":
        return {"function_name": "none", "arguments": {}, "reason": chart_plan.get("reason")}

    if not isinstance(arguments, dict):
        raise ValueError("Chart arguments must be a JSON object.")

    column_arguments = {
        "bar_chart": ["x", "y"],
        "line_chart": ["x", "y"],
        "pie_chart": ["names", "values"],
        "scatter_chart": ["x", "y"],
    }

    required_arguments = column_arguments[function_name] + ["title"]
    missing_arguments = [
        argument_name
        for argument_name in required_arguments
        if not arguments.get(argument_name)
    ]

    if missing_arguments:
        raise ValueError(f"Chart plan is missing required argument(s): {missing_arguments}")

    missing_columns = [
        arguments.get(argument_name)
        for argument_name in column_arguments[function_name]
        if not column_exists(sql_result, arguments.get(argument_name))
    ]

    if missing_columns:
        raise ValueError(f"Chart plan references missing column(s): {missing_columns}")

    color_column = arguments.get("color")
    if color_column and not column_exists(sql_result, color_column):
        raise ValueError(f"Chart plan references missing color column: {color_column}")

    return chart_plan


def bar_chart(data, x, y, title, color=None, x_title=None, y_title=None):
    fig = px.bar(data, x=x, y=y, color=color, title=title)
    fig.update_layout(xaxis_title=x_title or x, yaxis_title=y_title or y)
    return style_chart(fig)


def line_chart(data, x, y, title, color=None, x_title=None, y_title=None):
    fig = px.line(data, x=x, y=y, color=color, markers=True, title=title)
    fig.update_layout(xaxis_title=x_title or x, yaxis_title=y_title or y)
    return style_chart(fig)


def pie_chart(data, names, values, title):
    fig = px.pie(data, names=names, values=values, title=title, hole=0.35)
    return style_chart(fig)


def scatter_chart(data, x, y, title, color=None, x_title=None, y_title=None):
    fig = px.scatter(data, x=x, y=y, color=color, title=title)
    fig.update_layout(xaxis_title=x_title or x, yaxis_title=y_title or y)
    return style_chart(fig)


def style_chart(fig):
    fig.update_layout(
        template="plotly_white",
        title_x=0.02,
        margin={"l": 20, "r": 20, "t": 70, "b": 30},
        legend_title_text="",
        height=440,
    )
    return fig


def build_chart_from_plan(sql_result, chart_plan):
    function_name = chart_plan["function_name"]

    if function_name == "none":
        return None

    chart_functions = {
        "bar_chart": bar_chart,
        "line_chart": line_chart,
        "pie_chart": pie_chart,
        "scatter_chart": scatter_chart,
    }

    return chart_functions[function_name](sql_result, **chart_plan["arguments"])


def save_chart_html(fig):
    CHARTS_DIR.mkdir(exist_ok=True)
    chart_path = CHARTS_DIR / "latest_chart.html"
    fig.write_html(chart_path, include_plotlyjs="cdn")
    return chart_path


def generate_chart_for_result(user_question, chart_hint, sql_result, model_name=None):
    model_name = model_name or get_default_model_name()
    with traced_span(
        "plan-and-render-chart",
        input={
            "user_question": user_question,
            "chart_hint": chart_hint,
            "result_row_count": len(sql_result) if isinstance(sql_result, list) else None,
        },
    ) as span:
        if not is_chart_requested(chart_hint) or not is_chartable_result(sql_result):
            output = {
                "chart_created": False,
                "reason": "Chart was not requested or SQL result is not chartable.",
            }
            safe_update_observation(span, output=output)
            return None, None, None

        chart_prompt = create_chart_prompt(user_question, chart_hint, sql_result)
        chart_response_text = gemini_call(model_name, chart_prompt)
        chart_plan = validate_chart_plan(load_string_as_json(chart_response_text), sql_result)
        logger.info("Chart agent plan: %s", chart_plan)

        fig = build_chart_from_plan(sql_result, chart_plan)
        chart_path = save_chart_html(fig) if fig else None

        safe_update_observation(
            span,
            output={
                "chart_created": fig is not None,
                "chart_plan": chart_plan,
                "chart_path": str(chart_path) if chart_path else None,
            },
        )
        return fig, chart_plan, chart_path
