import mediapipe as mp
import cv2
import numpy as np

class Detector:
    def __init__(self, max_faces=50):
        import os
        import urllib.request
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        
        models = {
            'face_landmarker.task': 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task',
            'hand_landmarker.task': 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task',
            'efficientdet_lite0.tflite': 'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite',
            'blaze_face_short_range.tflite': 'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite'
        }
        for file, url in models.items():
            if not os.path.exists(file):
                print(f"Downloading {file}...")
                urllib.request.urlretrieve(url, file)

        base_options_f = python.BaseOptions(model_asset_path='face_landmarker.task')
        options_f = vision.FaceLandmarkerOptions(
            base_options=base_options_f,
            num_faces=max_faces,
            min_face_detection_confidence=0.4,
            min_face_presence_confidence=0.4,
            min_tracking_confidence=0.4)
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options_f)
        
        base_options_h = python.BaseOptions(model_asset_path='hand_landmarker.task')
        options_h = vision.HandLandmarkerOptions(
            base_options=base_options_h,
            num_hands=max_faces * 2)
        self.hand_landmarker = vision.HandLandmarker.create_from_options(options_h)

        base_options_fd = python.BaseOptions(model_asset_path='blaze_face_short_range.tflite')
        options_fd = vision.FaceDetectorOptions(base_options=base_options_fd, min_detection_confidence=0.5)
        self.face_detector = vision.FaceDetector.create_from_options(options_fd)

        base_options_o = python.BaseOptions(model_asset_path='efficientdet_lite0.tflite')
        options_o = vision.ObjectDetectorOptions(base_options=base_options_o, score_threshold=0.2)
        self.object_detector = vision.ObjectDetector.create_from_options(options_o)

    def process(self, frame):
        import mediapipe as mp
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        face_results = self.face_landmarker.detect(mp_image)
        hand_results = self.hand_landmarker.detect(mp_image)
        object_results = self.object_detector.detect(mp_image)
        fd_results = self.face_detector.detect(mp_image)
        return face_results, hand_results, object_results, fd_results

    def extract_persons(self, object_results):
        persons = []
        if hasattr(object_results, 'detections') and object_results.detections:
            for detection in object_results.detections:
                category = detection.categories[0]
                if category.category_name == 'person':
                    bbox = detection.bounding_box
                    persons.append({
                        'bbox': [bbox.origin_x, bbox.origin_y, bbox.origin_x + bbox.width, bbox.origin_y + bbox.height, category.score]
                    })
        return persons

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

    def extract_short_range_faces(self, fd_results, existing_faces):
        # Identify any faces blaze_face caught that landmarker missed
        if hasattr(fd_results, 'detections') and fd_results.detections:
            for detection in fd_results.detections:
                bbox = detection.bounding_box
                x1, y1 = bbox.origin_x, bbox.origin_y
                x2, y2 = x1 + bbox.width, y1 + bbox.height
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                
                # Check for overlap with existing
                covered = False
                for ef in existing_faces:
                    efx1, efy1, efx2, efy2, _ = ef['bbox']
                    if efx1 <= cx <= efx2 and efy1 <= cy <= efy2:
                        covered = True
                        break
                        
                if not covered:
                    score = detection.categories[0].score if detection.categories else 1.0
                    existing_faces.append({
                        'bbox': [x1, y1, x2, y2, score],
                        'landmarks': None
                    })
        return existing_faces
