import cv2
import numpy as np
from fer.fer import FER
import mediapipe as mp

class Analyzer:
    def __init__(self):
        # Using MTCNN=False to speed up FER since we already have the face crop
        self.emotion_detector = FER(mtcnn=False)

    def get_head_pose(self, landmarks, frame_shape):
        # 3D model points (Y axis flipped to match image coordinate system)
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip 1
            (0.0, 330.0, -65.0),         # Chin 152
            (-225.0, -170.0, -135.0),    # Left eye left corner 33
            (225.0, -170.0, -135.0),     # Right eye right corner 263
            (-150.0, 150.0, -125.0),     # Left Mouth corner 61
            (150.0, 150.0, -125.0)       # Right mouth corner 291
        ])
        
        # 2D image points from landmarks
        h, w, c = frame_shape
        image_points = np.array([
            (landmarks[1].x * w, landmarks[1].y * h),
            (landmarks[152].x * w, landmarks[152].y * h),
            (landmarks[33].x * w, landmarks[33].y * h),
            (landmarks[263].x * w, landmarks[263].y * h),
            (landmarks[61].x * w, landmarks[61].y * h),
            (landmarks[291].x * w, landmarks[291].y * h)
        ], dtype="double")
        
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )
        dist_coeffs = np.zeros((4, 1))

        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs)

        rmat, jac = cv2.Rodrigues(rotation_vector)
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
        
        x = angles[0] # pitch
        y = angles[1] # yaw
        z = angles[2] # roll

        # Simple classification
        if y < -20:
            pose = "Left"
        elif y > 20:
            pose = "Right"
        elif x < -15:
            pose = "Down"
        elif x > 20:
            pose = "Up"
        else:
            pose = "Forward"

        return pose, (x, y, z)

    def get_ear(self, landmarks, frame_shape):
        h, w = frame_shape[:2]
        # calculate eye aspect ratio to detect drowsiness
        def calculate_ear(eye_indices):
            p1 = np.array([landmarks[eye_indices[0]].x * w, landmarks[eye_indices[0]].y * h])
            p2 = np.array([landmarks[eye_indices[1]].x * w, landmarks[eye_indices[1]].y * h])
            p3 = np.array([landmarks[eye_indices[2]].x * w, landmarks[eye_indices[2]].y * h])
            p4 = np.array([landmarks[eye_indices[3]].x * w, landmarks[eye_indices[3]].y * h])
            p5 = np.array([landmarks[eye_indices[4]].x * w, landmarks[eye_indices[4]].y * h])
            p6 = np.array([landmarks[eye_indices[5]].x * w, landmarks[eye_indices[5]].y * h])

            ear = (np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2.0 * np.linalg.norm(p1 - p4))
            return ear

        left_ear = calculate_ear([33, 160, 158, 133, 153, 144])
        right_ear = calculate_ear([362, 385, 387, 263, 373, 380])
        return (left_ear + right_ear) / 2.0

    def get_emotion(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox[:4])
        h, w = frame.shape[:2]
        
        # Add slight margin out so FER captures forehead/chin properly
        margin_y = int(0.25 * (y2 - y1))
        margin_x = int(0.15 * (x2 - x1))
        c_x1 = max(0, x1 - margin_x)
        c_y1 = max(0, y1 - margin_y)
        c_x2 = min(w, x2 + margin_x)
        c_y2 = min(h, y2 + margin_y)
        
        face_img = frame[c_y1:c_y2, c_x1:c_x2]
        
        if face_img.size == 0 or face_img.shape[0] < 10 or face_img.shape[1] < 10:
            return "neutral", {"neutral": 1.0}

        try:
            emotion, score = self.emotion_detector.top_emotion(face_img)
            if emotion is None:
                return "neutral", {"neutral": 1.0}
                
            emotions_dict = self.emotion_detector.detect_emotions(face_img)
            if len(emotions_dict) > 0:
                scores = emotions_dict[0]['emotions']
                return emotion, scores
            return emotion, {emotion: score}
        except Exception as e:
            return "neutral", {"neutral": 1.0}

    def check_hand_raised(self, bbox, hand_results, frame_shape):
        h, w, c = frame_shape
        x1, y1, x2, y2 = map(int, bbox[:4])
        
        if not hasattr(hand_results, 'hand_landmarks') or not hand_results.hand_landmarks:
            return False
            
        face_center_x = (x1 + x2) / 2
        for hand_landmarks in hand_results.hand_landmarks:
            wrist_y = hand_landmarks[0].y * h
            finger_y = hand_landmarks[12].y * h
            hand_x = hand_landmarks[12].x * w
            
            # Check if hand is raised (finger above wrist) and is near the person horizontally and vertically
            if finger_y < wrist_y and finger_y < y1 and abs(hand_x - face_center_x) < (x2 - x1) * 3:
                return True
        return False
