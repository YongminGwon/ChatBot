from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-large-v3")
AutoProcessor.from_pretrained("openai/whisper-large-v3")