import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options_o = python.BaseOptions(model_asset_path='c:/Users/mahid/Desktop/Programming stuff/Python stuff/Lecture_lens/efficientdet_lite0.tflite')
options_o = vision.ObjectDetectorOptions(base_options=base_options_o, score_threshold=0.1)
detector = vision.ObjectDetector.create_from_options(options_o)

print("Model loaded successfully")
# let's just inspect the model output if possible, or print something about categories
