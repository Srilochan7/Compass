import torch
import sounddevice as sd
from transformers import pipeline
import time

# --- Config ---
MODEL_NAME = "distil-whisper/distil-large-v2"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
DURATION_SECONDS = 5
SAMPLE_RATE = 16000

def record_audio():
    print("Get ready to speak...")
    for i in range(3, 0, -1):
        print(i)
        time.sleep(1)

    print("Recording...")
    audio = sd.rec(
        int(DURATION_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"   # pipeline expects float32
    )
    sd.wait()
    print("Recording finished.")
    return audio.squeeze()  # remove extra dimension

def load_transcriber():
    print(f"Loading model: {MODEL_NAME}")
    transcriber = pipeline(
        "automatic-speech-recognition",
        model=MODEL_NAME,
        device=DEVICE,
        torch_dtype=TORCH_DTYPE
    )
    print("Model loaded.")
    return transcriber

if __name__ == "__main__":
    audio = record_audio()
    transcriber = load_transcriber()

    print("\nTranscribing...")
    result = transcriber({"array": audio, "sampling_rate": SAMPLE_RATE})
    print("\n--- Transcription ---")
    print(result["text"])



def record_and_transcribe(duration=10):
    global DURATION_SECONDS
    DURATION_SECONDS = duration  # override default if needed
    
    # 1. Record audio
    audio = record_audio()
    
    # 2. Load model
    transcriber = load_transcriber()
    
    # 3. Transcribe
    result = transcriber({"array": audio, "sampling_rate": SAMPLE_RATE})
    
    # 4. Return the text
    return result["text"]
