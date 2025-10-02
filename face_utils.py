import os
import pickle
import numpy as np
from deepface import DeepFace

# --- Configuration ---
KNOWN_FACES_DIR = "faces"
MODEL_NAME = "VGG-Face"
DETECTOR_BACKEND = 'opencv'
DISTANCE_METRIC = 'cosine'
RECOGNITION_THRESHOLD = 0.40
ENCODINGS_CACHE_PATH = "face_encodings.pkl"

# --- Global Lists for Known Faces ---
known_face_encodings = []
known_face_names = []

def has_directory_changed():
    """Checks if the faces directory has been modified since the last cache."""
    try:
        cache_mod_time = os.path.getmtime(ENCODINGS_CACHE_PATH)
        for root, _, files in os.walk(KNOWN_FACES_DIR):
            for file in files:
                if os.path.getmtime(os.path.join(root, file)) > cache_mod_time:
                    return True
    except FileNotFoundError:
        return True # Cache doesn't exist
    return False

def precompute_known_faces():
    """
    Analyzes all photos in the KNOWN_FACES_DIR and stores their
    face embeddings. Caches the results to a file for fast loading.
    """
    global known_face_encodings, known_face_names

    if not has_directory_changed():
        print("Loading face encodings from cache...")
        try:
            with open(ENCODINGS_CACHE_PATH, "rb") as f:
                cached_data = pickle.load(f)
                known_face_encodings = cached_data["encodings"]
                known_face_names = cached_data["names"]
                print("✅ Embeddings loaded from cache.")
                return
        except Exception as e:
            print(f"⚠️ Could not load cache, re-computing faces. Error: {e}")

    print("Re-computing embeddings for known faces... This may take a moment.")
    temp_encodings = []
    temp_names = []

    if not os.path.exists(KNOWN_FACES_DIR):
        print(f"ERROR: Directory not found: {KNOWN_FACES_DIR}")
        return

    for name in os.listdir(KNOWN_FACES_DIR):
        person_dir = os.path.join(KNOWN_FACES_DIR, name)
        if not os.path.isdir(person_dir): continue
        for filename in os.listdir(person_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(person_dir, filename)
                try:
                    embedding_objs = DeepFace.represent(img_path=image_path, model_name=MODEL_NAME, enforce_detection=True, detector_backend=DETECTOR_BACKEND)
                    temp_encodings.append(embedding_objs[0]["embedding"])
                    temp_names.append(name)
                    print(f"✅ Pre-computed embedding for {name} from {filename}")
                except (ValueError, IndexError):
                    print(f"⚠️ No face found in {filename}, skipping.")
                except Exception as e:
                    print(f"❌ An unexpected error occurred with {filename}: {e}")
    
    known_face_encodings = temp_encodings
    known_face_names = temp_names

    # Save to cache
    try:
        with open(ENCODINGS_CACHE_PATH, "wb") as f:
            pickle.dump({"encodings": known_face_encodings, "names": known_face_names}, f)
        print("✅ Face encodings saved to cache.")
    except Exception as e:
        print(f"❌ Could not save embeddings to cache. Error: {e}")


def find_best_match(live_embedding):
    """
    Compares a live face embedding to all pre-computed known embeddings.
    Returns the name and distance of the best match.
    """
    if not known_face_encodings:
        return "INTRUDER", float('inf')

    distances = []
    live_emb_np = np.array(live_embedding)
    
    for known_emb in known_face_encodings:
        known_emb_np = np.array(known_emb)
        dist = 1 - (np.dot(live_emb_np, known_emb_np) / (np.linalg.norm(live_emb_np) * np.linalg.norm(known_emb_np)))
        distances.append(dist)

    if not distances:
        return "INTRUDER", float('inf')

    min_distance_index = np.argmin(distances)
    min_distance = distances[min_distance_index]

    if min_distance <= RECOGNITION_THRESHOLD:
        return known_face_names[min_distance_index], min_distance
    else:
        return "INTRUDER", min_distance


# import cv2
# import os
# import time
# import numpy as np
# from deepface import DeepFace

# # --- Configuration ---
# KNOWN_FACES_DIR = "faces"
# MODEL_NAME = "VGG-Face"
# DETECTOR_BACKEND = 'opencv'
# DISTANCE_METRIC = 'cosine'
# # The threshold for a match. Lower is stricter. Cosine similarity is usually around 0.40.
# # If you're getting false positives, make it lower (e.g., 0.35).
# # If it's failing to recognize you, make it higher (e.g., 0.50).
# RECOGNITION_THRESHOLD = 0.50
# CAMERA_WARMUP_TIME = 2.0

# # --- Global Lists for Known Faces ---
# known_face_encodings = []
# known_face_names = []

# def precompute_known_faces():
#     """
#     Analyzes all photos in the KNOWN_FACES_DIR at startup and stores their
#     face embeddings in memory for fast comparison later.
#     """
#     print("Pre-computing embeddings for known faces... This may take a moment.")
#     if not os.path.exists(KNOWN_FACES_DIR):
#         print(f"ERROR: Directory not found: {KNOWN_FACES_DIR}")
#         return

#     for name in os.listdir(KNOWN_FACES_DIR):
#         person_dir = os.path.join(KNOWN_FACES_DIR, name)
#         if not os.path.isdir(person_dir):
#             continue

#         for filename in os.listdir(person_dir):
#             if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
#                 image_path = os.path.join(person_dir, filename)
#                 try:
#                     # represent() finds the face and computes the embedding
#                     embedding_objs = DeepFace.represent(
#                         img_path=image_path,
#                         model_name=MODEL_NAME,
#                         enforce_detection=True, # This will error if no face is found
#                         detector_backend=DETECTOR_BACKEND
#                     )
#                     # We only care about the first face found in the photo
#                     embedding = embedding_objs[0]["embedding"]
#                     known_face_encodings.append(embedding)
#                     known_face_names.append(name)
#                     print(f"✅ Pre-computed embedding for {name} from {filename}")
#                 except ValueError:
#                     # This error is thrown by DeepFace when no face is detected
#                     print(f"⚠️ No face found in {filename}, skipping.")
#                 except Exception as e:
#                     print(f"❌ An unexpected error occurred with {filename}: {e}")

# def find_best_match(live_embedding):
#     """
#     Compares a live face embedding to all pre-computed known embeddings.
#     Returns the name and distance of the best match.
#     """
#     if not known_face_encodings:
#         return "INTRUDER", float('inf')

#     distances = []
#     # Using numpy for efficient calculations
#     live_emb_np = np.array(live_embedding)
    
#     for known_emb in known_face_encodings:
#         known_emb_np = np.array(known_emb)
#         # Calculate cosine distance
#         dist = 1 - (np.dot(live_emb_np, known_emb_np) / (np.linalg.norm(live_emb_np) * np.linalg.norm(known_emb_np)))
#         distances.append(dist)

#     if not distances:
#         return "INTRUDER", float('inf')

#     min_distance_index = np.argmin(distances)
#     min_distance = distances[min_distance_index]

#     if min_distance <= RECOGNITION_THRESHOLD:
#         return known_face_names[min_distance_index], min_distance
#     else:
#         return "INTRUDER", min_distance

# def main():
#     """Main function to run face recognition from the webcam using DeepFace."""
#     precompute_known_faces()
#     if not known_face_encodings:
#         print("!!! FATAL ERROR: No known faces were loaded. Exiting. !!!")
#         return

#     print("\nStarting webcam...")
#     video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
#     time.sleep(CAMERA_WARMUP_TIME)

#     if not video_capture.isOpened():
#         print("!!! FATAL: Could not open webcam. Is it being used by another application? !!!")
#         return

#     print("Webcam started. Looking for faces... (Press 'q' to quit)")

#     while True:
#         ret, frame = video_capture.read()
#         if not ret:
#             print("Error: Failed to grab frame from webcam.")
#             break

#         try:
#             # This finds all faces in the frame and computes their embeddings in one go.
#             live_embedding_objs = DeepFace.represent(
#                 img_path=frame,
#                 model_name=MODEL_NAME,
#                 enforce_detection=False,
#                 detector_backend=DETECTOR_BACKEND
#             )

#             for live_obj in live_embedding_objs:
#                 live_embedding = live_obj["embedding"]
                
#                 # --- THE BUG FIX: Safely get coordinates by key name ---
#                 facial_area = live_obj["facial_area"]
#                 x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
                
#                 name, distance = find_best_match(live_embedding)
                
#                 box_color = (0, 0, 255) if name == "INTRUDER" else (0, 255, 0)
                
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
#                 cv2.rectangle(frame, (x, y - 35), (x + w, y), box_color, cv2.FILLED)
#                 font = cv2.FONT_HERSHEY_DUPLEX
#                 cv2.putText(frame, name, (x + 6, y - 6), font, 1.0, (255, 255, 255), 1)

#         except Exception as e:
#             # This will now only catch unexpected errors, not the bug from before
#             # print(f"An error occurred in the main loop: {e}") # Uncomment for debugging
#             pass

#         cv2.imshow('Face Recognition (DeepFace) - Press Q to Quit', frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     video_capture.release()
#     cv2.destroyAllWindows()
#     print("Program terminated.")

# if __name__ == "__main__":
#     main()

