import setup_env
setup_env.ensure_environment()
import cv2
import time
from modules.detection import Detector

detector = Detector()
cap = cv2.VideoCapture(0)

print("Starting analysis...", flush=True)

for i in range(10):
    ret, frame = cap.read()
    if not ret: continue
    
    frame = cv2.flip(frame, 1)
    face_res, hand_res, obj_res = detector.process(frame)
    
    faces = detector.extract_faces(face_res, frame.shape)
    persons = detector.extract_persons(obj_res)
    
    print(f"Frame {i}:", flush=True)
    print(f"  Faces: {len(faces)}", flush=True)
    for f in faces:
        print(f"    - box: {f['bbox']}", flush=True)
    print(f"  Persons: {len(persons)}", flush=True)
    for p in persons:
        print(f"    - box: {p['bbox']}", flush=True)
    time.sleep(0.5)

cap.release()
print("Done.", flush=True)
