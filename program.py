import pyaudio
import speech_recognition as sr
import wave
import threading
import cv2
import sys
import time
import os
import io
import numpy as np
from playsound import playsound
from dotenv import load_dotenv
from collections import deque
from deepface import DeepFace
from scipy.spatial import distance as dist
from face_utils import precompute_known_faces, find_best_match
from llm_handler import generate_escalation_dialogue
from intent import load_intent_model, predict_intent

load_dotenv()

ACTIVATION_KEYWORDS = ["protect", "room"]
DEACTIVATION_KEYWORDS = ["stop", "protecting"]
CAMERA_WARMUP_TIME = 2.0
FRAMES_TO_PROCESS_PER_SECOND = 2  # two frames per second in the webcam for better performance withoout hanging
MODEL_NAME = "VGG-Face"
DETECTOR_BACKEND = 'opencv'
INTRUDER_PERSISTENCE_FRAMES = 3
ESCALATION_DELAY_SECONDS = 8
FORMAT = pyaudio.paInt16
CHANNELS = 1; RATE = 16000; CHUNK = 1024; RECORD_SECONDS = 4
guard_mode_active = False
stop_listening_event = threading.Event()
last_command = deque(maxlen=1)
last_user_response = deque(maxlen=1)
next_object_id = 0
objects = {}
object_dossiers = {}
last_known_faces = [] # Store the last detected face info to draw between processing frames

class Intruder:
    def __init__(self, object_id):
        self.id = object_id
        self.escalation_level = 0
        self.last_interaction_time = 0
        self.name = "INTRUDER"

    def should_escalate_on_timer(self):
        return time.time() - self.last_interaction_time > ESCALATION_DELAY_SECONDS

    def increment_level_and_update_time(self):
        if self.escalation_level == 0:
             self.escalation_level = 1
        else:
            self.escalation_level = min(self.escalation_level + 1, 3)
        self.last_interaction_time = time.time()

def speak(audio_file):
    # Plays the generated audio file and then deletes it.
    try:
        print(f"▶️ Playing audio...")
        playsound(audio_file)
        os.remove(audio_file)
    except Exception as e:
        print(f"❌ Could not play audio: {e}")

def recognize_speech(wav_data):
    # Transcribes audio data using Google's Web Speech API.
    recognizer = sr.Recognizer()
    with sr.AudioFile(io.BytesIO(wav_data)) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
        return text.lower().strip()
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
        return None
    except sr.RequestError as e:
        print(f"Could not request results from Google service; {e}")
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
    
    transcription = recognize_speech(wav_bytes)

    if transcription:
        print(f'Heard: "{transcription}"')
        is_activation = all(word in transcription for word in ACTIVATION_KEYWORDS)
        is_deactivation = all(word in transcription for word in DEACTIVATION_KEYWORDS)
        if is_activation and not guard_mode_active:
            last_command.append('activate')
        elif is_deactivation and guard_mode_active:
            last_command.append('deactivate')
        elif guard_mode_active:
            last_user_response.append(transcription)

def continuous_listening_thread():
    while guard_mode_active and not stop_listening_event.is_set(): listen_and_process_command(is_guarding=True)
    print("Deactivation listener has stopped.")

def start_guard_mode():
    global guard_mode_active, stop_listening_event, next_object_id, objects, object_dossiers, last_user_response, last_known_faces
    
    guard_mode_active = True; stop_listening_event.clear(); last_command.clear(); last_user_response.clear()
    next_object_id = 0; objects = {}; object_dossiers = {}

    print("\n====================================================")
    print("=== ✅ GUARD MODE ACTIVATED - WEBCAM IS LIVE ===")
    print(f"=== (Say '{' '.join(DEACTIVATION_KEYWORDS)}' to deactivate)      ===")
    print("====================================================\n")

    listener_thread = threading.Thread(target=continuous_listening_thread, daemon=True); listener_thread.start()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW); print("Initializing webcam..."); time.sleep(CAMERA_WARMUP_TIME)
    
    if not cap.isOpened():
        print("Error: Could not open webcam."); guard_mode_active = False; return
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30 
    frame_skip_interval = max(1, int(fps / FRAMES_TO_PROCESS_PER_SECOND))
    last_known_faces.clear()

    while guard_mode_active:
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1
        process_this_frame = (frame_count % frame_skip_interval == 0)

        if process_this_frame:
            try:
                # This block now only runs periodically
                current_faces_info = []
                # Use a copy of the frame for processing to avoid race conditions
                processing_frame = frame.copy()

                live_objs = DeepFace.represent(img_path=processing_frame, model_name=MODEL_NAME, enforce_detection=False, detector_backend=DETECTOR_BACKEND)
                rects = [(obj['facial_area']['x'], obj['facial_area']['y'], obj['facial_area']['w'], obj['facial_area']['h']) for obj in live_objs]
                input_centroids = np.array([(x + w // 2, y + h // 2) for x, y, w, h in rects], dtype="int")
                
                if len(objects) == 0:
                    for i in range(len(input_centroids)):
                        objects[next_object_id] = input_centroids[i]
                        object_dossiers[next_object_id] = {'id': next_object_id, 'history': deque(maxlen=INTRUDER_PERSISTENCE_FRAMES), 'handler': None}
                        next_object_id += 1
                else:
                    object_ids = list(objects.keys()); object_centroids = list(objects.values())
                    if len(input_centroids) > 0:
                        D = dist.cdist(np.array(object_centroids), input_centroids)
                        rows = D.min(axis=1).argsort(); cols = D.argmin(axis=1)[rows]
                        used_rows, used_cols = set(), set()
                        for (row, col) in zip(rows, cols):
                            if row in used_rows or col in used_cols: continue
                            object_id = object_ids[row]; objects[object_id] = input_centroids[col]; used_rows.add(row); used_cols.add(col)
                        
                        unused_rows = set(range(D.shape[0])).difference(used_rows)
                        unused_cols = set(range(D.shape[1])).difference(used_cols)

                        if D.shape[0] >= len(input_centroids):
                            for row in unused_rows:
                                object_id = object_ids[row]
                                if object_id in objects: del objects[object_id]
                                if object_id in object_dossiers: del object_dossiers[object_id]
                        
                        for col in unused_cols:
                            objects[next_object_id] = input_centroids[col]
                            object_dossiers[next_object_id] = {'id': next_object_id, 'history': deque(maxlen=INTRUDER_PERSISTENCE_FRAMES), 'handler': None}
                            next_object_id += 1
                    else: # No faces detected in this frame, clear old objects
                        objects.clear()
                        object_dossiers.clear()


                for live_obj in live_objs:
                    centroid = (live_obj['facial_area']['x'] + live_obj['facial_area']['w'] // 2, live_obj['facial_area']['y'] + live_obj['facial_area']['h'] // 2)
                    object_id_for_face = next((oid for oid, c in objects.items() if np.array_equal(c, centroid)), None)
                    if object_id_for_face is None or object_id_for_face not in object_dossiers: continue

                    name, _ = find_best_match(live_obj["embedding"])
                    
                    object_dossiers[object_id_for_face]['history'].append(name)
                    history = object_dossiers[object_id_for_face]['history']
                    stable_name = max(set(history), key=history.count) if history else "INTRUDER"
                    
                    if stable_name == "INTRUDER" and len(history) == INTRUDER_PERSISTENCE_FRAMES and all(h == "INTRUDER" for h in history):
                        handler = object_dossiers[object_id_for_face]['handler']
                        if handler is None:
                            handler = Intruder(object_id_for_face)
                            object_dossiers[object_id_for_face]['handler'] = handler

                        user_input = last_user_response.popleft() if last_user_response else None
                        should_speak = False
                        
                        if user_input:
                            should_speak = True; handler.last_interaction_time = time.time()
                        elif handler.should_escalate_on_timer():
                            should_speak = True; handler.increment_level_and_update_time()

                        if should_speak:
                            audio_file = generate_escalation_dialogue(handler.id, handler.escalation_level, user_input=user_input)
                            if audio_file:
                                speak(audio_file)

                    facial_area = live_obj['facial_area']
                    x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
                    box_color = (0, 0, 255) if stable_name == "INTRUDER" else (0, 255, 0)
                    
                    current_faces_info.append({
                        "box": (x, y, w, h),
                        "name": stable_name,
                        "color": box_color
                    })

                last_known_faces = current_faces_info # Update the list for drawing

            except Exception as e:
                print(f"❌ Error in main loop: {e}")
        
        # --- Drawing logic now runs EVERY frame, using the last known data ---
        for face_info in last_known_faces:
            x, y, w, h = face_info["box"]
            stable_name = face_info["name"]
            box_color = face_info["color"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
            cv2.rectangle(frame, (x, y - 35), (x + w, y), box_color, cv2.FILLED)
            cv2.putText(frame, stable_name, (x + 6, y - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

        cv2.imshow('Security Feed - Press Q to Quit', frame)
        try:
            if last_command.popleft() == 'deactivate': guard_mode_active = False
        except IndexError: pass
        if cv2.waitKey(1) & 0xFF == ord('q'): guard_mode_active = False
    
    stop_listening_event.set(); cap.release(); cv2.destroyAllWindows(); cv2.waitKey(1)
    listener_thread.join(timeout=1.0); print("\n--- 🛑 GUARD MODE DEACTIVATED ---")

def main():
    global guard_mode_active, stop_listening_event

    if not os.getenv("GOOGLE_API_KEY"):
        print("!!! FATAL ERROR: GOOGLE_API_KEY not found in .env file. !!!")
        sys.exit(1)
    
    precompute_known_faces()
    print("🚀 Security Agent Initialized.\n")
    model, tokenizer, label_encoder = load_intent_model()

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
            print("\nExiting program...");
            guard_mode_active = False;
            stop_listening_event.set();
            break

if __name__ == "__main__":
    main()