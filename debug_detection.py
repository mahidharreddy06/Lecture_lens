import setup_env
setup_env.ensure_environment()
import cv2
import time
from modules.detection import Detector

detector = Detector()
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

time.sleep(1) # wait for camera to warmup

for i in range(3):
    ret, frame = cap.read()
    if not ret: continue
    
    frame = cv2.flip(frame, 1)
    print(f"\n--- Frame {i} ---")
    print(f"Frame shape: {frame.shape}")
    
    face_results, hand_results, object_results, fd_results = detector.process(frame)
    
    faces = detector.extract_faces(face_results, frame.shape)
    print("FaceLandmarker faces:")
    for f in faces: print("  ", f['bbox'])
    
    print("BlazeFace short range raw:")
    if hasattr(fd_results, 'detections') and fd_results.detections:
        for d in fd_results.detections:
            print(f"  bbox: x={d.bounding_box.origin_x}, y={d.bounding_box.origin_y}, w={d.bounding_box.width}, h={d.bounding_box.height}, score={d.categories[0].score}")
    else:
        print("  None")
        
    faces = detector.extract_short_range_faces(fd_results, faces)
    print("Combined faces:")
    for f in faces: print("  ", f['bbox'])

    persons = detector.extract_persons(object_results)
    print("EfficientDet persons:")
    for p in persons: print("  ", p['bbox'])

cap.release()
