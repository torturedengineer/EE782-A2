import pyaudio
import requests
import wave
import threading
import cv2
import sys
import time
import os
import io
import numpy as np
import pyttsx3
from dotenv import load_dotenv
from collections import deque
from deepface import DeepFace
from scipy.spatial import distance as dist

# --- Import our custom modules ---
from face_utils import precompute_known_faces, find_best_match
from llm_handler import generate_escalation_dialogue

# --- Load Environment Variables ---
load_dotenv()

# --- Configuration ---
API_KEY = os.getenv("HUGGING_FACE_API_KEY") # For Whisper
WHISPER_API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"
ACTIVATION_KEYWORDS = ["guard", "room"]
DEACTIVATION_KEYWORDS = ["stop", "guarding"]
CAMERA_WARMUP_TIME = 2.0

# Intruder Management Config
INTRUDER_PERSISTENCE_FRAMES = 10
ESCALATION_DELAY_SECONDS = 15

# Audio recording parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1; RATE = 16000; CHUNK = 1024; RECORD_SECONDS = 4

# --- Global State & Tracking ---
guard_mode_active = False
stop_listening_event = threading.Event()
last_command = deque(maxlen=1)
tts_engine = None
next_object_id = 0
objects = {}
object_dossiers = {}

class Intruder:
    # (This class is unchanged)
    def __init__(self, object_id):
        self.id = object_id; self.escalation_level = 0; self.last_warning_time = 0; self.name = "INTRUDER"
    def escalate(self):
        current_time = time.time()
        if current_time - self.last_warning_time > ESCALATION_DELAY_SECONDS:
            self.escalation_level = min(self.escalation_level + 1, 3)
            self.last_warning_time = current_time
            return True
        return False

# (TTS, Whisper, and listening functions are mostly unchanged)
def initialize_tts():
    global tts_engine
    try: tts_engine = pyttsx3.init(); print("✅ TTS Engine Initialized.")
    except Exception as e: print(f"❌ Could not initialize TTS engine: {e}")

def speak(text):
    if tts_engine: print(f"🤖 AGENT SAYS: {text}"); tts_engine.say(text); tts_engine.runAndWait()
    else: print(f"🤖 AGENT (TTS not working): {text}")

def query_whisper(audio_data):
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "audio/wav"}
    for _ in range(3):
        try:
            response = requests.post(WHISPER_API_URL, headers=headers, data=audio_data, timeout=25)
            if response.status_code == 200: return response.json().get('text', '').lower().strip()
            print(f"Whisper API Error: Status {response.status_code}, Response: {response.text}")
            return None
        except requests.exceptions.RequestException as e: print(f"Network error: {e}"); time.sleep(1)
    return None

def listen_and_process_command(is_guarding):
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    prefix = "🚨" if is_guarding else "🎤"; print(f"{prefix} Listening...")
    frames = [stream.read(CHUNK, exception_on_overflow=False) for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS))]
    stream.stop_stream(); stream.close(); audio.terminate()
    print("...Processing Audio...")
    with io.BytesIO() as audio_buffer:
        with wave.open(audio_buffer, 'wb') as wf:
            wf.setnchannels(CHANNELS); wf.setsampwidth(audio.get_sample_size(FORMAT)); wf.setframerate(RATE); wf.writeframes(b''.join(frames))
        wav_bytes = audio_buffer.getvalue()
    transcription = query_whisper(wav_bytes)
    if transcription:
        print(f'Whisper heard: "{transcription}"')
        is_activation = all(word in transcription for word in ACTIVATION_KEYWORDS)
        is_deactivation = all(word in transcription for word in DEACTIVATION_KEYWORDS)
        if is_activation and not guard_mode_active: last_command.append('activate')
        elif is_deactivation and guard_mode_active: last_command.append('deactivate')

def continuous_listening_thread():
    while guard_mode_active and not stop_listening_event.is_set(): listen_and_process_command(is_guarding=True)
    print("Deactivation listener has stopped.")

# --- Main Guard Mode Logic (Now uses LLM) ---

def start_guard_mode():
    guard_mode_active, stop_listening_event, next_object_id, objects, object_dossiers
    guard_mode_active = True; stop_listening_event.clear(); last_command.clear()
    next_object_id = 0; objects = {}; object_dossiers = {}

    print("\n====================================================")
    print("=== ✅ GUARD MODE ACTIVATED - WEBCAM IS LIVE ===")
    print(f"=== (Say '{' '.join(DEACTIVATION_KEYWORDS)}' to deactivate)      ===")
    print("====================================================\n")

    listener_thread = threading.Thread(target=continuous_listening_thread, daemon=True); listener_thread.start()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW); print("Initializing webcam..."); time.sleep(CAMERA_WARMUP_TIME)
    
    if not cap.isOpened():
        print("Error: Could not open webcam."); guard_mode_active = False; return

    while guard_mode_active:
        ret, frame = cap.read()
        if not ret: break
        try:
            live_objs = DeepFace.represent(img_path=frame, model_name=MODEL_NAME, enforce_detection=False, detector_backend=DETECTOR_BACKEND)
            rects = [tuple(obj['facial_area'].values()) for obj in live_objs]
            input_centroids = np.array([(x + w // 2, y + h // 2) for x, y, w, h in rects], dtype="int")
            
            # (Centroid tracking logic remains the same)
            if len(objects) == 0:
                for i in range(len(input_centroids)):
                    objects[next_object_id] = input_centroids[i]
                    object_dossiers[next_object_id] = {'id': next_object_id, 'history': deque(maxlen=INTRUDER_PERSISTENCE_FRAMES), 'handler': None}
                    next_object_id += 1
            else:
                object_ids = list(objects.keys()); object_centroids = list(objects.values())
                D = dist.cdist(np.array(object_centroids), input_centroids)
                rows = D.min(axis=1).argsort(); cols = D.argmin(axis=1)[rows]
                used_rows, used_cols = set(), set()
                for (row, col) in zip(rows, cols):
                    if row in used_rows or col in used_cols: continue
                    object_id = object_ids[row]; objects[object_id] = input_centroids[col]; used_rows.add(row); used_cols.add(col)
                unused_rows = set(range(D.shape[0])).difference(used_rows); unused_cols = set(range(D.shape[1])).difference(used_cols)
                for row in unused_rows: object_id = object_ids[row]; del objects[object_id]; del object_dossiers[object_id]
                for col in unused_cols:
                    objects[next_object_id] = input_centroids[col]
                    object_dossiers[next_object_id] = {'id': next_object_id, 'history': deque(maxlen=INTRUDER_PERSISTENCE_FRAMES), 'handler': None}
                    next_object_id += 1

            for live_obj in live_objs:
                centroid = (live_obj['facial_area']['x'] + live_obj['facial_area']['w'] // 2, live_obj['facial_area']['y'] + live_obj['facial_area']['h'] // 2)
                object_id_for_face = next((oid for oid, c in objects.items() if tuple(c) == centroid), None)
                if object_id_for_face is None: continue

                name, _ = find_best_match(live_obj["embedding"])
                object_dossiers[object_id_for_face]['history'].append(name)
                history = object_dossiers[object_id_for_face]['history']
                stable_name = max(set(history), key=history.count) if history else "INTRUDER"
                
                # --- LLM ESCALATION LOGIC ---
                if stable_name == "INTRUDER" and len(history) == INTRUDER_PERSISTENCE_FRAMES and all(h == "INTRUDER" for h in history):
                    handler = object_dossiers[object_id_for_face]['handler']
                    if handler is None:
                        handler = Intruder(object_id_for_face)
                        object_dossiers[object_id_for_face]['handler'] = handler
                    
                    if handler.escalate():
                        dialogue = generate_escalation_dialogue(handler.escalation_level)
                        speak(dialogue)

                x,y,w,h = live_obj['facial_area'].values()
                box_color = (0, 0, 255) if stable_name == "INTRUDER" else (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
                cv2.rectangle(frame, (x, y - 35), (x + w, y), box_color, cv2.FILLED)
                cv2.putText(frame, stable_name, (x + 6, y - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
        except Exception as e: pass

        cv2.imshow('Security Feed - Press Q to Quit', frame)
        try:
            if last_command.popleft() == 'deactivate': guard_mode_active = False
        except IndexError: pass
        if cv2.waitKey(1) & 0xFF == ord('q'): guard_mode_active = False
    
    stop_listening_event.set(); cap.release(); cv2.destroyAllWindows(); cv2.waitKey(1)
    listener_thread.join(timeout=1.0); print("\n--- 🛑 GUARD MODE DEACTIVATED ---")

def main():
    if not API_KEY or not os.getenv("GOOGLE_API_KEY"):
        print("!!! FATAL ERROR: API key(s) not found in .env file. !!!")
        print("Please ensure both HUGGING_FACE_API_KEY and GOOGLE_API_KEY are set in your .env file.")
        sys.exit(1)
    
    initialize_tts(); precompute_known_faces()
    print("🚀 Security Agent Initialized.\n")

    while True:
        try:
            if not guard_mode_active:
                print("--- WAITING FOR ACTIVATION ---")
                print(f"(Say '{' '.join(ACTIVATION_KEYWORDS)} ...')")
                listen_and_process_command(is_guarding=False)
                try:
                    if last_command.popleft() == 'activate': start_guard_mode()
                except IndexError: pass
            time.sleep(0.05) 
        except KeyboardInterrupt:
            print("\nExiting program..."); guard_mode_active; guard_mode_active = False; stop_listening_event.set(); break

if __name__ == "__main__":
    main()
