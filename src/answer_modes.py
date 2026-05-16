ANSWER_MODE_EXECUTIVE = "executive_summary"
ANSWER_MODE_ANALYST = "analyst_detail"
ANSWER_MODE_AUDIT = "audit_evidence"
ANSWER_MODE_SQL_DEBUG = "sql_debug"

DEFAULT_ANSWER_MODE = ANSWER_MODE_ANALYST


ANSWER_MODES = {
    ANSWER_MODE_EXECUTIVE: {
        "key": ANSWER_MODE_EXECUTIVE,
        "label": "Executive summary",
        "description": "Lead with the answer, confidence, and edge-case notes.",
    },
    ANSWER_MODE_ANALYST: {
        "key": ANSWER_MODE_ANALYST,
        "label": "Analyst detail",
        "description": "Show the analyst view with results, charts, evidence, and concise SQL context.",
    },
    ANSWER_MODE_AUDIT: {
        "key": ANSWER_MODE_AUDIT,
        "label": "Audit evidence",
        "description": "Lead with assumptions, evidence trail, validation, and limitations.",
    },
    ANSWER_MODE_SQL_DEBUG: {
        "key": ANSWER_MODE_SQL_DEBUG,
        "label": "SQL/debug",
        "description": "Expose generated SQL, raw payloads, cache, clarification, and entity metadata.",
    },
}


ANSWER_MODE_LABEL_TO_KEY = {
    mode["label"]: key
    for key, mode in ANSWER_MODES.items()
}


def normalize_answer_mode(answer_mode):
    if not answer_mode:
        return DEFAULT_ANSWER_MODE

    if answer_mode in ANSWER_MODES:
        return answer_mode

    return ANSWER_MODE_LABEL_TO_KEY.get(str(answer_mode), DEFAULT_ANSWER_MODE)


def answer_mode_label(answer_mode):
    return ANSWER_MODES[normalize_answer_mode(answer_mode)]["label"]


def answer_mode_options():
    return list(ANSWER_MODES.keys())


def answer_mode_metadata(answer_mode):
    mode = ANSWER_MODES[normalize_answer_mode(answer_mode)]
    return {
        "answer_mode": mode["key"],
        "answer_mode_label": mode["label"],
        "answer_mode_description": mode["description"],
    }


def apply_answer_mode_to_sql_output(sql_output, answer_mode):
    output = dict(sql_output or {})
    mode_metadata = answer_mode_metadata(answer_mode)

    output["Answer_Mode"] = mode_metadata["answer_mode"]
    output["Answer_Mode_Label"] = mode_metadata["answer_mode_label"]
    output.setdefault("Metadata", {})
    if isinstance(output["Metadata"], dict):
        output["Metadata"].update(mode_metadata)

    analysis = output.get("Analysis")
    if isinstance(analysis, dict):
        analysis = dict(analysis)
        analysis["Answer_Mode"] = mode_metadata["answer_mode"]
        analysis["Answer_Mode_Label"] = mode_metadata["answer_mode_label"]
        analysis.setdefault("Metadata", {})
        if isinstance(analysis["Metadata"], dict):
            analysis["Metadata"].update(mode_metadata)
        output["Analysis"] = analysis

    return output
