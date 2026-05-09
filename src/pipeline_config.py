from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SEMANTIC_LAYER_PATH = ROOT_DIR / "data" / "semantic_layer.json"
MAX_MEMORY_TURNS = 6
MAX_MEMORY_SQL_CHARS = 1600
MAX_MEMORY_FIELD_CHARS = 600
RESULT_SAMPLE_ROWS = 3
MAX_CLARIFICATION_ATTEMPTS = 1
INSUFFICIENT_DATA_PREFIX = "insufficient data in context:"
