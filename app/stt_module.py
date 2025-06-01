import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import os

_stt_pipe = None

def init_stt_model():
    global _stt_pipe
    if _stt_pipe is not None:
        print("STT 파이프라인이 이미 초기화되어 있습니다.")
        return
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = os.path.join(os.path.dirname(__file__), "models", "whisper-large-v3")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    _stt_pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )
    print("STT 파이프라인 초기화 완료.")

def transcribe_audio(audio_path: str) -> str | None:
    global _stt_pipe
    if _stt_pipe is None:
        print("STT 파이프라인이 초기화되지 않았습니다. 먼저 init_stt_model()을 호출하세요.")
        return None
    try:
        result = _stt_pipe(audio_path, generate_kwargs={"language": "ko"})
        return result["text"]
    except Exception as e:
        print(f"STT 변환 오류: {e}")
        return None
