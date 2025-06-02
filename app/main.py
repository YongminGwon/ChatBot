import os
import webrtcvad
import pyaudio
import collections
import time
from dotenv import load_dotenv
load_dotenv()

from stt_module import init_stt_model, transcribe_audio
from textgen_module import textgen

FORMAT = pyaudio.paInt16  # 오디오 포맷 (16-bit PCM)
CHANNELS = 1              # 모노 채널
RATE = 16000              # 샘플링 레이트 (Hz). webrtcvad는 8000, 16000, 32000, 48000 Hz 지원
FRAME_DURATION_MS = 30    # VAD 프레임 지속 시간 (10, 20, 또는 30 ms)
SAMPLES_PER_FRAME = int(RATE * FRAME_DURATION_MS / 1000)  # VAD 프레임 당 샘플 수
BYTES_PER_SAMPLE = pyaudio.get_sample_size(FORMAT) # 샘플 당 바이트 수 (paInt16는 2)
BYTES_PER_FRAME = SAMPLES_PER_FRAME * BYTES_PER_SAMPLE # VAD 프레임 당 바이트 수

VAD_AGGRESSIVENESS = 3
SPEECH_FRAMES_TRIGGER = 3
SILENCE_FRAMES_TRIGGER = 25

def make_pyaudio():
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=SAMPLES_PER_FRAME)
    return audio, stream

def make_vad():
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    return vad

def make_buffer():
    return collections.deque(maxlen=SILENCE_FRAMES_TRIGGER)

def save_audio_data(audio_data, filename):
    """오디오 데이터를 WAV 파일로 저장"""
    import wave
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(BYTES_PER_SAMPLE)
        wf.setframerate(RATE)
        wf.writeframes(b''.join(audio_data))

def process_speech(audio_data):
    """음성 데이터를 처리하고 AI 응답 생성"""
    # 임시 파일로 저장
    temp_file = "temp_audio.wav"
    save_audio_data(audio_data, temp_file)
    
    # STT 처리
    user_text = transcribe_audio(temp_file)
    if not user_text:
        print("STT 변환 실패.")
        return None
    elif user_text == "MBC 뉴스 김성현입니다.":
        return None

    print(f"STT 결과: {user_text}")
    
    # AI 응답 생성
    ai_response = textgen(user_text)
    print(f"AI 응답: {ai_response}")
    
    # 임시 파일 삭제
    os.remove(temp_file)
    return ai_response

def main():
    print("AI 챗봇을 시작합니다.")
    init_stt_model()  # STT 모델 초기화

    # PyAudio 및 VAD 초기화
    audio, stream = make_pyaudio()
    vad = make_vad()
    
    # 버퍼 초기화
    speech_start_buffer = collections.deque(maxlen=SPEECH_FRAMES_TRIGGER)
    silence_end_buffer = collections.deque(maxlen=SILENCE_FRAMES_TRIGGER)
    
    triggered = False
    speech_audio_data = []

    print(f"샘플링 레이트: {RATE} Hz")
    print(f"프레임 지속 시간: {FRAME_DURATION_MS} ms")
    print(f"프레임 당 샘플 수: {SAMPLES_PER_FRAME}")
    print(f"프레임 당 바이트 수: {BYTES_PER_FRAME}")
    print(f"VAD 민감도: {VAD_AGGRESSIVENESS}")
    print(f"말 끝 판단 정적 프레임 수: {SILENCE_FRAMES_TRIGGER} (약 {SILENCE_FRAMES_TRIGGER * FRAME_DURATION_MS / 1000:.2f} 초)")
    print("-" * 30)
    print("대기 중... (Ctrl+C 로 종료)")

    try:
        while True:
            audio_frame_bytes = stream.read(SAMPLES_PER_FRAME, exception_on_overflow=False)

            if len(audio_frame_bytes) < BYTES_PER_FRAME:
                time.sleep(0.01)
                continue

            try:
                is_speech = vad.is_speech(audio_frame_bytes, RATE)
            except Exception as e:
                continue

            if not triggered:
                speech_start_buffer.append(is_speech)
                if sum(speech_start_buffer) >= SPEECH_FRAMES_TRIGGER:
                    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                    print(f"[{timestamp}] === 말 시작 ===")
                    triggered = True
                    speech_audio_data.append(audio_frame_bytes)
                    silence_end_buffer.clear()
                    silence_end_buffer.append(is_speech)
            else:
                silence_end_buffer.append(is_speech)
                speech_audio_data.append(audio_frame_bytes)

                if len(silence_end_buffer) == SILENCE_FRAMES_TRIGGER and sum(silence_end_buffer) == 0:
                    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                    print(f"[{timestamp}] === 말 끝 === (녹음된 프레임 수: {len(speech_audio_data)})")

                    # 음성 처리 및 AI 응답
                    process_speech(speech_audio_data)

                    triggered = False
                    speech_start_buffer.clear()
                    speech_audio_data = []

    except KeyboardInterrupt:
        print("\n종료 중...")
    finally:
        print("스트림 및 PyAudio 객체 정리 중...")
        stream.stop_stream()
        stream.close()
        audio.terminate()
        print("정리 완료.")

if __name__ == "__main__":
    main()