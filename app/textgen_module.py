from google import genai
from google.genai import types
import time
import os

API_KEY_FROM_ENV = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY_FROM_ENV)

SYSTEM_INSTRUCTION = "말하는 새우, 팝송을 좋아한다, 피자랑 맥주를 같이먹는다, 바다새우인데 민물로 산책을 간다, 이모티콘을 쓰지 않음, 기호도 쓰지 않음"

def textgen(prompt: str, max_retries=3, initial_delay=1) -> str | None:
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                config=types.GenerateContentConfig(temperature=1.5, system_instruction=SYSTEM_INSTRUCTION),
                contents=[prompt]
            )
            return response.text
        except Exception as e:
            if attempt == max_retries - 1:  # 마지막 시도였다면
                print(f"API 호출 실패: {str(e)}")
                return "죄송합니다. 서버가 일시적으로 과부하 상태입니다. 잠시 후 다시 시도해주세요."
            
            delay = initial_delay * (2 ** attempt)  # 지수 백오프
            print(f"API 호출 실패. {delay}초 후 재시도합니다...")
            time.sleep(delay)