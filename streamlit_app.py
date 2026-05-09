import importlib.util
import json
import sys
from pathlib import Path

import streamlit as st


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
SQL_RUNNER_PATH = SRC_DIR / "02_run_sql_on_sqlite.py"

sys.path.append(str(SRC_DIR))

from pipeline import generate_sql_for_question
from chart_agent import generate_chart_for_result


def load_sql_runner():
    spec = importlib.util.spec_from_file_location("sql_runner", SQL_RUNNER_PATH)
    sql_runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sql_runner)
    return sql_runner


def run_pipeline(user_question):
    sql_output = generate_sql_for_question(user_question)
    generated_sql = sql_output.get("SQL")

    if not generated_sql:
        return sql_output, None, None, None, None, None

    sql_runner = load_sql_runner()
    sql_result = sql_runner.run_query(generated_sql)

    if isinstance(sql_result, str):
        return sql_output, sql_result, None, None, None, None

    try:
        fig, chart_plan, chart_path = generate_chart_for_result(
            user_question=user_question,
            chart_hint=sql_output.get("Chart"),
            sql_result=sql_result,
        )
        chart_error = None
    except Exception as e:
        fig, chart_plan, chart_path = None, None, None
        chart_error = str(e)

    return sql_output, sql_result, fig, chart_plan, chart_path, chart_error


def is_empty_value(value):
    return value is None or value == "" or value == [] or value == {}


def show_sql_generation_output(sql_output):
    st.subheader("SQL Generation Output")

    non_sql_keys = [key for key in sql_output.keys() if key != "SQL"]
    if not non_sql_keys:
        st.info("No non-SQL generation fields were returned.")
        return

    for key in non_sql_keys:
        value = sql_output.get(key)
        st.markdown(f"**{key}**")

        if is_empty_value(value):
            st.info("Empty")
        elif isinstance(value, (dict, list)):
            st.json(value)
        else:
            st.write(value)


st.set_page_config(page_title="NL to SQL", layout="wide")

st.title("NL to SQL")

user_question = st.text_area(
    "Enter your question",
    placeholder="Example: How many invoices were raised last month?",
    height=120,
)

run_button = st.button("Run Pipeline", type="primary")

if run_button:
    if not user_question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating SQL, running query, and planning chart..."):
            try:
                sql_output, sql_result, fig, chart_plan, chart_path, chart_error = run_pipeline(user_question.strip())
            except Exception as e:
                st.error(f"Pipeline failed: {e}")
                st.stop()

        show_sql_generation_output(sql_output)

        generated_sql = sql_output.get("SQL")
        if generated_sql:
            st.subheader("Generated SQL")
            st.code(generated_sql, language="sql")

        st.subheader("SQL Running Result")
        if sql_result is None:
            st.info("SQL was not generated, so query execution was skipped.")
        elif isinstance(sql_result, str):
            st.error(sql_result)
        else:
            st.dataframe(sql_result, use_container_width=True)

        if fig is not None:
            st.subheader("Chart")
            st.plotly_chart(fig, use_container_width=True)

            if chart_plan:
                with st.expander("Chart Agent Plan"):
                    st.json(chart_plan)

            if chart_path:
                st.caption(f"Stored chart: {chart_path}")
        elif chart_error:
            st.warning(f"Chart generation skipped: {chart_error}")

        with st.expander("Raw SQL Running Result"):
            st.code(json.dumps(sql_result, indent=2, default=str), language="json")
