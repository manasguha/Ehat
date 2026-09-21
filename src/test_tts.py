"""Test Kokoro TTS (PyTorch version)"""
import os
import soundfile as sf
import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

from kokoro import KPipeline

print("Initializing KPipeline...")
pipeline = KPipeline(lang_code='a')  # American English

text = "Electric vehicle charging stations experience changing electrical demand throughout the day. Our project uses machine learning to predict station load."
print("Generating audio...")
generator = pipeline(text, voice='af_heart', speed=1.0)

out_dir = r"C:\Users\manas\clg\ehat\charging-station-load-prediction\outputs"
os.makedirs(out_dir, exist_ok=True)

for i, (gs, ps, audio) in enumerate(generator):
    out_path = os.path.join(out_dir, f"test_kokoro_{i}.wav")
    sf.write(out_path, audio, 24000)
    duration = len(audio) / 24000
    print(f"  Segment {i}: {duration:.2f}s - {gs[:50]}...")

print("TTS test complete!")
