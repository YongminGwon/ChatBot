from google import genai
from google.genai import types
import os

API_KEY_FROM_ENV = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY_FROM_ENV)

def textgen(prompt: str) -> str | None:
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(temperature=1.5, system_instruction=prompt),
        contents=[prompt]
    )
    return response.text