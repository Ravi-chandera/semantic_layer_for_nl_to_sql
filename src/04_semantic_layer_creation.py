from google import genai
from dotenv import load_dotenv
import os
from prompt import SEMANTIC_LAYER_PROMPT

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client()


def gemini_call(model_name, contents):

    response = client.models.generate_content(
        model=model_name, contents=contents
    )
    return response.text

def create_semantic_layer(model_name, contents):
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
    model_name = "gemini-3-flash-preview"
    schema_json = load_schema_json("../data/schema.json")
    contents = SEMANTIC_LAYER_PROMPT + "\n\n" + schema_json
    create_semantic_layer(model_name, contents)