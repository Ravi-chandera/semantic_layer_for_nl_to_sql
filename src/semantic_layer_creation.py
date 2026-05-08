from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client()


def gemini_call(model_name, contents):

    response = client.models.generate_content(
        model=model_name, contents=contents
    )
    return response.text

if __name__ == "__main__":
    model_name = "gemini-3-flash-preview"
    contents = "Explain how AI works in ogabuga kid language"
    print(gemini_call(model_name, contents))