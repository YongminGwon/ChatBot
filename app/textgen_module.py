from google import genai
from google.genai import types
import os

API_KEY_FROM_ENV = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY_FROM_ENV)

def textgen(prompt: str) -> str | None:
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(temperature=1.5, system_instruction="시니컬한 여자아이, 말을 길게 하지 않음,팩트 폭력(가끔), 효율성과 논리 중시, 아주 가끔 보여주는 의외의 모습"),
        contents=[prompt]
    )
    return response.text