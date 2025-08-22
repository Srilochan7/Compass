# import torch
# from transformers import pipeline
# import time

# # --- Configuration ---
# MODEL_NAME = "distil-whisper/distil-large-v2"
# DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
# TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# def load_transcriber():
#     """Loads the optimized distil-whisper pipeline onto the GPU."""
#     print(f"Loading model '{MODEL_NAME}' onto {DEVICE}...")
#     transcriber = pipeline(
#         "automatic-speech-recognition",
#         model=MODEL_NAME,
#         device=DEVICE,
#         torch_dtype=TORCH_DTYPE
#     )
#     print("Whisper model loaded successfully.")
#     return transcriber

# def transcribe_audio_file(transcriber, audio_file_path):
#     """Transcribes a given audio file and times the process."""
#     if not transcriber:
#         print("Transcriber not loaded.")
#         return

#     print(f"\nTranscribing audio file: {audio_file_path}...")
#     start_time = time.time()
    
#     result = transcriber(audio_file_path)
    
#     end_time = time.time()
#     print(f"Transcription finished in {end_time - start_time:.2f} seconds.")
    
#     return result["text"]

# if __name__ == "__main__":
#     # 1. Load the model (this happens only once)
#     speech_transcriber = load_transcriber()
    
#     # 2. Transcribe an audio file (replace with your file)
#     # You would call this function every time you get a new recording
#     audio_file = "path/to/your/symptom_recording.wav" 
#     transcription = transcribe_audio_file(speech_transcriber, audio_file)
    
#     if transcription:
#         print("\n--- Transcription Result ---")
#         print(transcription)