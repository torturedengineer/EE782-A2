#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
import requests
import pyaudio
import sys

# --- Configuration ---
WHISPER_API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"

# --- ⬇️ PASTE YOUR HUGGING FACE 'WRITE' API KEY HERE ⬇️ ---
# Get one from https://huggingface.co/settings/tokens
hf_api_key = "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"
# -----------------------------------------------------------

# Audio recording parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 4

def main():
    """Records audio, sends it to Whisper, and prints the result."""
    # --- API Key Check ---
    if hf_api_key == "hf_YOUR_WRITE_API_KEY_HERE" or not hf_api_key:
        print("!!! ERROR: Please edit the script (line 9) and paste your 'write' API key. !!!")
        sys.exit(1)

    # --- 1. Record Audio ---
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    
    print("🎤 Recording for 4 seconds... Please speak now.")
    frames = []
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
    
    print("...Finished recording. Processing audio...")
    stream.stop_stream()
    stream.close()
    audio.terminate()
    audio_data = b''.join(frames)

    # --- 2. Call API ---
    headers = {"Authorization": f"Bearer {hf_api_key}"}
    try:
        response = requests.post(WHISPER_API_URL, headers=headers, data=audio_data, timeout=20)
        
        if response.status_code == 200:
            result = response.json()
            transcription = result.get('text', 'No transcription found in response.')
            print("\n✅ SUCCESS!")
            print(f'Whisper heard: "{transcription}"')
        else:
            # Provide a clean error message
            print(f"\n❌ FAILED! API returned Status Code: {response.status_code}")
            try:
                error_details = response.json()
                print(f"Error Details: {error_details.get('error', 'No specific error message.')}")
            except ValueError:
                print("Could not parse error details from the server's response.")

    except requests.exceptions.RequestException as e:
        print(f"\n❌ FAILED! A network error occurred: {e}")

if __name__ == "__main__":
    main()
