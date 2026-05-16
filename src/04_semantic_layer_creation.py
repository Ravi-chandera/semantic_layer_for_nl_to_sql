from google import genai
try:
    from .model_config import get_default_model_name, require_gemini_api_key
    from .prompt import SEMANTIC_LAYER_PROMPT
except ImportError:
    from model_config import get_default_model_name, require_gemini_api_key
    from prompt import SEMANTIC_LAYER_PROMPT


def gemini_call(model_name, contents):
    api_key = require_gemini_api_key()
    model_name = model_name or get_default_model_name()
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model_name, contents=contents
    )
    return response.text

def create_semantic_layer(model_name, contents):
    api_key = require_gemini_api_key()
    model_name = model_name or get_default_model_name()
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model_name, contents=contents
    )
    with open("../data/semantic_layer.json", "w") as f:
        f.write(response.text)
    print("Semantic layer created successfully!")

def load_schema_json(file_path):
    with open(file_path, 'r') as f:
        return f.read()

if __name__ == "__main__":
    model_name = get_default_model_name()
    schema_json = load_schema_json("../data/schema.json")
    contents = SEMANTIC_LAYER_PROMPT + "\n\n" + schema_json
    create_semantic_layer(model_name, contents)
