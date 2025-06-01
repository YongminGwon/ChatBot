import os
from dotenv import load_dotenv
load_dotenv()

from stt_module import init_stt_model, transcribe_audio
from textgen_module import textgen

def answer_from_audio(audio_path):
    user_text = transcribe_audio(audio_path)
    if not user_text:
        print("STT 변환 실패.")
        return None
    print(f"STT 결과: {user_text}")
    ai_response = textgen(user_text)
    print(f"AI 응답: {ai_response}")
    return ai_response

if __name__ == "__main__":
    print("AI 챗봇을 시작합니다.")
    init_stt_model()  # 한 번만 초기화

    # 한 개의 오디오 파일만 처리 (예: audio/1.mp3)
    audio_path = os.path.join(os.path.dirname(__file__), "audio", "1.mp3")
    answer_from_audio(audio_path)
