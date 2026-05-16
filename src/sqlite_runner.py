import importlib.util
from functools import lru_cache
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
LEGACY_SQL_RUNNER_PATH = ROOT_DIR / "src" / "02_run_sql_on_sqlite.py"


@lru_cache(maxsize=1)
def _legacy_module():
    spec = importlib.util.spec_from_file_location(
        "src._legacy_sql_runner",
        LEGACY_SQL_RUNNER_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load SQL runner from {LEGACY_SQL_RUNNER_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_query(query, db_name=None):
    runner = _legacy_module()
    if db_name is None:
        return runner.run_query(query)
    return runner.run_query(query, db_name=db_name)


def __getattr__(name):
    return getattr(_legacy_module(), name)
