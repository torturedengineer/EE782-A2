import os
import pickle
import numpy as np
from deepface import DeepFace

KNOWN_FACES_DIR = "faces"
MODEL_NAME = "VGG-Face"
DETECTOR_BACKEND = 'opencv'
RECOGNITION_THRESHOLD = 0.50
ENCODINGS_CACHE_PATH = "face_encodings.pkl"

known_face_encodings = []
known_face_names = []

def precompute_known_faces():
    #first checks if empty, and if empty then computes the embeddings which is not what i want
    global known_face_encodings, known_face_names
    # if os.path.exists(ENCODINGS_CACHE_PATH):
    #     print("Loading face encodings from cache...")
    #     try:
    #         with open(ENCODINGS_CACHE_PATH, "rb") as f:
    #             cached_data = pickle.load(f)
    #             known_face_encodings = cached_data["encodings"]
    #             known_face_names = cached_data["names"]
    #             print("Embeddings loaded from cache.")
    #             return
    #     except Exception as e:
    #         print(f"Could not load cache, re-computing faces. Error: {e}")

    print("Computing embeddings for known faces... This may take a moment.")
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
                    print(f" Pre-computed embedding for {name} from {filename}")
                except (ValueError, IndexError):
                    print(f" No face found in {filename}, skipping.")
                except Exception as e:
                    print(f" An unexpected error occurred with {filename}: {e}")
    known_face_encodings = temp_encodings
    known_face_names = temp_names
    try:
        with open(ENCODINGS_CACHE_PATH, "wb") as f:
            pickle.dump({"encodings": known_face_encodings, "names": known_face_names}, f)
        print("✅ Face encodings saved to cache.")
    except Exception as e:
        print(f"❌ Could not save embeddings to cache. Error: {e}")

def find_best_match(live_embedding):
    """
    Compares a live face embedding to all pre-computed known embeddings.
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

# import os
# import pickle
# import numpy as np
# from deepface import DeepFace

# # --- Configuration ---
# KNOWN_FACES_DIR = "faces"
# MODEL_NAME = "VGG-Face"
# DETECTOR_BACKEND = 'opencv'
# RECOGNITION_THRESHOLD = 0.40
# ENCODINGS_CACHE_PATH = "face_encodings.pkl"

# # --- Global Lists for Known Faces ---
# known_face_encodings = []
# known_face_names = []

# def precompute_known_faces():
#     """
#     Analyzes all photos in the KNOWN_FACES_DIR and stores their
#     face embeddings. Caches the results to a file for fast loading.
#     """
#     global known_face_encodings, known_face_names

#     # --- SIMPLIFIED CACHING LOGIC ---
#     if os.path.exists(ENCODINGS_CACHE_PATH):
#         print("Loading face encodings from cache...")
#         try:
#             with open(ENCODINGS_CACHE_PATH, "rb") as f:
#                 cached_data = pickle.load(f)
#                 known_face_encodings = cached_data["encodings"]
#                 known_face_names = cached_data["names"]
#                 print("✅ Embeddings loaded from cache.")
#                 return
#         except Exception as e:
#             print(f"⚠️ Could not load cache, re-computing faces. Error: {e}")

#     print("Re-computing embeddings for known faces... This may take a moment.")
#     temp_encodings = []
#     temp_names = []

#     if not os.path.exists(KNOWN_FACES_DIR):
#         print(f"ERROR: Directory not found: {KNOWN_FACES_DIR}")
#         return

#     for name in os.listdir(KNOWN_FACES_DIR):
#         person_dir = os.path.join(KNOWN_FACES_DIR, name)
#         if not os.path.isdir(person_dir): continue
#         for filename in os.listdir(person_dir):
#             if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
#                 image_path = os.path.join(person_dir, filename)
#                 try:
#                     embedding_objs = DeepFace.represent(img_path=image_path, model_name=MODEL_NAME, enforce_detection=True, detector_backend=DETECTOR_BACKEND)
#                     temp_encodings.append(embedding_objs[0]["embedding"])
#                     temp_names.append(name)
#                     print(f"✅ Pre-computed embedding for {name} from {filename}")
#                 except (ValueError, IndexError):
#                     print(f"⚠️ No face found in {filename}, skipping.")
#                 except Exception as e:
#                     print(f"❌ An unexpected error occurred with {filename}: {e}")
    
#     known_face_encodings = temp_encodings
#     known_face_names = temp_names

#     try:
#         with open(ENCODINGS_CACHE_PATH, "wb") as f:
#             pickle.dump({"encodings": known_face_encodings, "names": known_face_names}, f)
#         print("✅ Face encodings saved to cache.")
#     except Exception as e:
#         print(f"❌ Could not save embeddings to cache. Error: {e}")

# def find_best_match(live_embedding):
#     """
#     Compares a live face embedding to all pre-computed known embeddings.
#     """
#     if not known_face_encodings:
#         return "INTRUDER", float('inf')

#     distances = []
#     live_emb_np = np.array(live_embedding)
    
#     for known_emb in known_face_encodings:
#         known_emb_np = np.array(known_emb)
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

