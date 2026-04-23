import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options_o = python.BaseOptions(model_asset_path='c:/Users/mahid/Desktop/Programming stuff/Python stuff/Lecture_lens/efficientdet_lite0.tflite')
options_o = vision.ObjectDetectorOptions(base_options=base_options_o, score_threshold=0.1)
detector = vision.ObjectDetector.create_from_options(options_o)

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if ret:
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    results = detector.detect(mp_image)
    print("Detections raw:", results)
    if hasattr(results, 'detections') and results.detections:
        for d in results.detections:
            print("Detected:", d.categories[0].category_name, d.categories[0].score, d.bounding_box)
    else:
        print("No detections")
else:
    print("Failed to read webcam")
cap.release()
