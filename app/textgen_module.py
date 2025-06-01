from google import genai
from google.genai import types
import os

API_KEY_FROM_ENV = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY_FROM_ENV)

SYSTEM_INSTRUCTION = "말하는 새우, 팝송을 좋아한다, 피자랑 맥주를 같이먹는다, 바다새우인데 민물로 산책을 간다, 이모티콘을 쓰지 않음, 기호도 쓰지 않음"

def textgen(prompt: str) -> str | None:
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(temperature=1.5, system_instruction=SYSTEM_INSTRUCTION),
        contents=[prompt]
    )
    return response.text