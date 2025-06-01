from transformers import AutoProcessor, BarkModel
import scipy
from pathlib import Path
import torch
import time
import os

# 모델 경로 설정
model_path = os.path.abspath("app/models/bark")

# 로컬 모델 로드
processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)

# 모델 로드 시 GPU 사용
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"선택된 device: {device}")
model = BarkModel.from_pretrained(model_path, local_files_only=True).to(device)

# 실제로 모델이 GPU에 올라갔는지 확인
if device == "cuda":
    try:
        first_param_device = next(model.parameters()).device
        print(f"모델이 올라간 device: {first_param_device}")
    except Exception:
        print("모델 파라미터 device 확인 불가 (BarkModel 구조에 따라 다를 수 있음)")
else:
    print("GPU를 사용하지 않습니다.")

voice_preset = "v2/ko_speaker_0"

inputs = processor(
    "반갑반갑 새우새우야.",
    voice_preset=voice_preset,
    return_tensors="pt"
)
inputs = {k: v.to(device) for k, v in inputs.items()}

# 생성 시간 측정 시작
start = time.time()
audio_array = model.generate(**inputs)
end = time.time()
print(f"음성 생성 시간: {end - start:.2f}초")

audio_array = audio_array.cpu().numpy().squeeze()
sample_rate = model.generation_config.sample_rate
scipy.io.wavfile.write("bark_out.wav", rate=sample_rate, data=audio_array)