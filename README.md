
## Architecture

The app follows a simple NL-to-SQL pipeline:

1. The Streamlit UI in `streamlit_app.py` accepts a natural language question.
2. `src/pipeline.py` loads `data/semantic_layer.json` and asks Gemini to route the question to relevant semantic-layer tables and metrics.
3. The same pipeline builds a focused SQL-generation prompt from the selected tables, metrics, join paths, ambiguity rules, and query hints.
4. Gemini returns a structured SQL response containing the generated SQL, explanation, assumptions, follow-up questions, and chart recommendation.
5. Before any SQL is executed, `src/02_run_sql_on_sqlite.py` runs regex-based SQL guardrails. The guardrail removes comments and quoted strings, then blocks generated SQL containing `INSERT`, `UPDATE`, or `DELETE`.
6. If the guardrail passes, the SQL runner validates referenced tables against the SQLite database, checks the query plan with `EXPLAIN QUERY PLAN`, and then executes the query on `data/assignment.db`.
7. The Streamlit UI displays the generated SQL and the query result.

This keeps write-operation protection at the SQL execution boundary, so any caller using `run_query()` gets the same guardrail before database access.

## Example video
sample input/output demonstrating the solution.

## DB Choice 

I have selected SQLite because my laptop will crash for production DBs. Also, most of LLMs are trained on spider dataset, a standard one for NL-to-SQL, it has SQLite SQLs, so they know syntax very well.

## AI Tool Usage

1. Prompt Enhancements 
I write whatever comes in my mind for a particular prompt and then tell Gemini to make it structured and fix any grammer issues.

2. Errors
I did fractional programming where I tell Gemini to give me specific code snippet or function. I put it and check it. I used Github Co-pilot to solve bugs that span over multiple files. 

3. Inline complition 
Copilot's inline compelation is pretty helpful while writing manual code.

## Self-reflections

Add where I would do different things or imporve if I have time and making it production ready.
