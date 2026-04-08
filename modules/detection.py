import mediapipe as mp
import cv2
import numpy as np

class Detector:
    def __init__(self, max_faces=10):
        import os
        import urllib.request
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        
        models = {
            'face_landmarker.task': 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task',
            'hand_landmarker.task': 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task'
        }
        for file, url in models.items():
            if not os.path.exists(file):
                print(f"Downloading {file}...")
                urllib.request.urlretrieve(url, file)

        base_options_f = python.BaseOptions(model_asset_path='face_landmarker.task')
        options_f = vision.FaceLandmarkerOptions(
            base_options=base_options_f,
            num_faces=max_faces)
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options_f)
        
        base_options_h = python.BaseOptions(model_asset_path='hand_landmarker.task')
        options_h = vision.HandLandmarkerOptions(
            base_options=base_options_h,
            num_hands=max_faces * 2)
        self.hand_landmarker = vision.HandLandmarker.create_from_options(options_h)

    def process(self, frame):
        import mediapipe as mp
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        face_results = self.face_landmarker.detect(mp_image)
        hand_results = self.hand_landmarker.detect(mp_image)
        return face_results, hand_results

    def extract_faces(self, face_results, frame_shape):
        h, w, _ = frame_shape
        faces = []
        if hasattr(face_results, 'face_landmarks') and face_results.face_landmarks:
            for face_landmarks in face_results.face_landmarks:
                x_min = w
                y_min = h
                x_max = 0
                y_max = 0
                for lm in face_landmarks:
                    x, y = int(lm.x * w), int(lm.y * h)
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)
                
                margin_y = int(0.2 * (y_max - y_min))
                margin_x = int(0.2 * (x_max - x_min))
                x_min = max(0, x_min - margin_x)
                y_min = max(0, y_min - margin_y)
                x_max = min(w, x_max + margin_x)
                y_max = min(h, y_max + margin_y)

                score = 1.0
                faces.append({
                    'bbox': [x_min, y_min, x_max, y_max, score],
                    'landmarks': face_landmarks
                })
        return faces
