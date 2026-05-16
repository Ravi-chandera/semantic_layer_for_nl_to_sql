# Reflection

## Why AI Is Essential

The core user is an AP investigation analyst, not a developer writing reports. Their questions are naturally incomplete, conversational, and business-specific:

- "Show me unpaid bills for Engineering."
- "Who are our top 5 vendors?"
- "Compare this quarter's invoice volume with last quarter."
- "For each vendor, show the running total of payments received."

AI is essential because the hard part is not only SQL syntax. The system has to infer analyst intent, map business terms to schema concepts, choose relevant metrics, resolve follow-ups, and decide when a question should not be answered without clarification. A static dashboard or report catalog can answer known questions, but AP investigations routinely combine vendors, departments, invoices, payments, products, dates, and risk flags in new ways.

The prototype uses AI where language and intent are genuinely variable, while keeping execution, validation, cache scoping, benchmark storage, and bridge-table expansion deterministic.

## Why Rules Alone Fail

A rule-only NL-to-SQL layer looks attractive at first because AP data has familiar entities: vendors, invoices, payments, purchase orders, departments, and products. It breaks down quickly:

| Rule-only pattern | Example | Failure mode |
| --- | --- | --- |
| Keyword matching | "bills" means `invoices` | Synonyms work for one word but not for full intent, such as unpaid bills by department with overdue filters. |
| Template library | "overdue invoices by vendor" | Every new combination of metric, join path, period, and grouping needs another template. |
| Default assumptions | "top vendors" means invoice value | The same phrase could mean invoice count, payment amount, outstanding balance, rating, or risk. Guessing creates misleading analysis. |
| Full-schema search | Send every table and column | The model or rules can select irrelevant joins and mix unrelated columns. |
| Fixed dashboard filters | Vendor, status, date | Follow-up investigation requires ad hoc joins and temporal comparisons that are not pre-modeled as screens. |

The implemented compromise is a semantic layer with deterministic guardrails plus AI generation. Rules provide known business facts and safety boundaries; AI handles composition and language variation.

## Architecture Tradeoffs

### Focused Semantic Context

I chose a router and context builder instead of sending the whole schema on every request. This reduces prompt noise and makes failures easier to diagnose, but it introduces a dependency on the router selecting enough context. To reduce that risk, the context builder deterministically adds bridge tables for known join paths.

### Clarification Before SQL

The system clarifies only when ambiguity materially changes the answer. This protects analyst trust for questions like "top vendors", but it can add friction. For AP investigation, a clarification is better than producing a confident but wrong vendor ranking.

### Structured JSON Output

The SQL generator returns fixed JSON fields for SQL, explanation, assumptions, follow-up question, and chart hint. This makes the app, cache, tests, and benchmark store more deterministic. The cost is a stricter prompt contract and extra normalization code for malformed model output.

### Semantic Cache

The cache improves repeated-question latency and reduces model calls. It is scoped by a semantic-layer hash so stale SQL is not reused after metadata changes. The tradeoff is operational complexity: ambiguous questions must bypass reuse, and embeddings can add local setup time.

### SQLite Safety Boundary

The prototype validates generated SQL before execution and blocks write operations. This is appropriate for a local demo, but production would need warehouse-level read-only credentials, timeouts, row limits, query-cost controls, and parser-backed validation.

## Evaluation Results

The deterministic test suite currently covers non-AI control flow and mocked model interactions:

```text
Ran 30 tests
OK
```

Covered behaviors include:

- clarification attempt tracking
- no-default ambiguity rules
- cache skipping when clarification is required
- bridge-table expansion
- executable-SQL response detection
- benchmark record persistence and dashboard generation
- mocked clarification-to-SQL flow that executes against SQLite

Manual verification focused on higher-risk AP analyst questions:

| Question | Result |
| --- | --- |
| Show me all invoices for the Engineering department. | Correctly used the `invoices -> purchase_orders -> departments` bridge path and executed. |
| For each vendor, show the running total of payments received. | Generated an executable window-function query over vendors, invoices, and payments. |
| What was our revenue last quarter? | Used the semantic `revenue` metric and previous-quarter date logic. |
| Who are our top 5 vendors? | Asked for clarification instead of silently choosing a ranking metric. |

## Failure Cases Found

The most important failure cases were not syntax errors; they were semantic failures:

- **Missing bridge joins**: department questions can require moving from invoices through purchase orders before reaching departments.
- **Metric ambiguity**: "top vendors" is unsafe without knowing whether top means invoice value, count, payments, outstanding balance, or rating.
- **Cache risk**: a previously generated answer for an ambiguous phrasing should not be reused before the clarification gate.
- **Window-query complexity**: running totals and previous-invoice comparisons need SQL patterns that simple keyword rules do not express reliably.
- **Model output variance**: generated text must be normalized and validated before any SQL is trusted.

## Changes After Testing

Testing and manual verification drove these changes:

- Added deterministic bridge-table expansion so department and payment paths have the required intermediate tables.
- Added no-default ambiguity handling so ambiguous AP questions ask a follow-up instead of guessing.
- Skipped semantic-cache reuse when a matched clarification rule requires user input.
- Added benchmark persistence and dashboard generation so evaluation runs are inspectable and repeatable.
- Added deterministic tests around clarification, cache behavior, SQL-response normalization, and benchmark storage.
- Documented the no-key path so reviewers can validate deterministic behavior without a Gemini key.

## Non-AI Baseline

| Capability | Non-AI baseline | AI-assisted prototype |
| --- | --- | --- |
| Simple counts and filters | Works with hand-written SQL or dashboard filters. | Works, while also explaining the result and recording benchmark metadata. |
| Synonyms | Requires a maintained synonym dictionary. | Uses semantic-layer synonyms plus model interpretation. |
| Multi-hop joins | Requires one template per join pattern. | Uses selected context plus deterministic bridge-table expansion. |
| Ambiguity detection | Usually guesses or returns no result. | Clarifies when the business answer would materially change. |
| Follow-up questions | Requires custom state machines. | Uses conversation memory to rewrite follow-ups into standalone questions. |
| New investigation paths | Requires new templates or dashboard work. | Can compose new read-only SQL within the semantic-layer boundary. |

## Remaining Work

- Add a larger golden-query evaluation set with expected SQL patterns and result assertions.
- Add a CLI benchmark runner in addition to the Streamlit tab.
- Harden SQL validation against the target production warehouse dialect.
- Add row limits, query timeouts, and production read-only credentials.
- Add a semantic-layer review workflow for AP domain owners.
- Track benchmark pass/fail expectations directly instead of only storing execution metadata.
