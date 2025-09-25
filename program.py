import pyaudio
import requests
import wave
import threading
import cv2
import sys
import time
import os
import io
from dotenv import load_dotenv
from collections import deque

# --- Load Environment Variables ---
load_dotenv()

# --- Configuration ---
API_KEY = os.getenv("HUGGING_FACE_API_KEY")
WHISPER_API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"
ACTIVATION_COMMAND = "guard my room"
DEACTIVATION_COMMAND = "stop guarding"

# Audio recording parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 4

# --- Global State Variables ---
guard_mode_active = False
stop_listening_event = threading.Event()
last_command = deque(maxlen=1)


def query_whisper(audio_data):
    """Sends audio data to Whisper API and returns the transcription."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "audio/wav"
    }
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(WHISPER_API_URL, headers=headers, data=audio_data, timeout=25)
            if response.status_code == 200:
                return response.json().get('text', '').lower().strip()
            else:
                print(f"Whisper API Error: Status {response.status_code}")
                try:
                    print(f"Response: {response.json().get('error')}")
                except requests.exceptions.JSONDecodeError:
                    print("Response: Not a valid JSON response from server.")
                return None
        except requests.exceptions.RequestException as e:
            print(f"A network error occurred: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {2 ** attempt} seconds...")
                time.sleep(2 ** attempt)
            else:
                print("Max retries reached. Failed to connect to API.")
                return None
    return None


def listen_and_process_command(is_guarding):
    """Records audio, packages it as a valid WAV file, and processes commands."""
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    prefix = "🚨" if is_guarding else "🎤"
    print(f"{prefix} Listening...")
    frames = []
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    audio.terminate()
    print("...Processing Audio...")

    audio_data_in_memory = io.BytesIO()
    with wave.open(audio_data_in_memory, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
    wav_bytes = audio_data_in_memory.getvalue()

    transcription = query_whisper(wav_bytes)
    if transcription:
        print(f'Whisper heard: "{transcription}"')
        activation_words = ACTIVATION_COMMAND.split()
        deactivation_words = DEACTIVATION_COMMAND.split()
        is_activation = all(word in transcription for word in activation_words)
        is_deactivation = all(word in transcription for word in deactivation_words)
        if is_activation and not guard_mode_active:
            print("--> Activation keywords detected!")
            last_command.append('activate')
        elif is_deactivation and guard_mode_active:
            print("--> Deactivation keywords detected!")
            last_command.append('deactivate')
    else:
        print("Could not understand audio or silence detected.")


def continuous_listening_thread():
    """A thread that continuously listens for the deactivation command."""
    while guard_mode_active and not stop_listening_event.is_set():
        listen_and_process_command(is_guarding=True)
    print("Deactivation listener has stopped.")


def start_guard_mode():
    """Activates guard mode, starting webcam and listening for deactivation."""
    global guard_mode_active, stop_listening_event
    guard_mode_active = True
    stop_listening_event.clear()
    last_command.clear()
    
    print("\n====================================================")
    print("=== ✅ GUARD MODE ACTIVATED - WEBCAM IS LIVE ===")
    print("=== (Say 'stop guarding' to deactivate)      ===")
    print("====================================================\n")

    listener_thread = threading.Thread(target=continuous_listening_thread, daemon=True)
    listener_thread.start()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    # --- NEW FIX: Add a short delay to allow the camera to initialize ---
    print("Initializing webcam...")
    time.sleep(2) # Give the camera 2 seconds to warm up
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        print("TRY THIS: Is another app using the camera? Or try changing the index from 0 to 1 in the code.")
        guard_mode_active = False
        return

    while guard_mode_active:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        cv2.imshow('Security Feed - Press Q to Quit', frame)
        try:
            if last_command.popleft() == 'deactivate':
                guard_mode_active = False
        except IndexError:
            pass
        if cv2.waitKey(1) & 0xFF == ord('q'):
            guard_mode_active = False
    
    stop_listening_event.set()
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    listener_thread.join(timeout=1.0)
    
    print("\n----------------------------------------------------")
    print("--- 🛑 GUARD MODE DEACTIVATED ---")
    print("----------------------------------------------------\n")


def main():
    """Main loop to wait for activation command."""
    global guard_mode_active, stop_listening_event
    if not API_KEY or "YOUR_WRITE_API_KEY_HERE" in API_KEY:
        print("!!! FATAL ERROR: HUGGING_FACE_API_KEY not found or not set in .env file. !!!")
        print("Please create a .env file and add your key, like this:")
        print('HUGGING_FACE_API_KEY="hf_..."')
        sys.exit(1)
    print("🚀 Security Agent Initialized.\n")
    while True:
        try:
            if not guard_mode_active:
                print("----------------------------------------------------")
                print("--- WAITING FOR ACTIVATION COMMAND ---")
                print("(Say 'guard my room')")
                listen_and_process_command(is_guarding=False)
                try:
                    if last_command.popleft() == 'activate':
                        start_guard_mode()
                except IndexError:
                    pass
            time.sleep(0.05) 
        except KeyboardInterrupt:
            print("\nExiting program. Goodbye!")
            guard_mode_active = False
            stop_listening_event.set()
            break

if __name__ == "__main__":
    main()

