import cv2
import math
import os
from modules.detection import Detector
from modules.tracking import Tracker
from modules.analysis import Analyzer
from modules.state import StateManager
from modules.ui import UI, mouse_callback
from modules.export import Exporter
from modules.gemini import AIAnalyzer

def main():
    detector = Detector()
    tracker = Tracker()
    analyzer = Analyzer()
    state_manager = StateManager()
    ui = UI(state_manager)
    exporter = Exporter(state_manager)
    gemini_analyzer = AIAnalyzer()

    # Webcam
    cap = cv2.VideoCapture(0)
    
    cv2.namedWindow('LectureLens')
    cv2.setMouseCallback('LectureLens', mouse_callback, {
        'state_manager': state_manager,
        'tracked_faces': []
    })

    print("LectureLens started. Press 'Q' to quit, 'E' to export report.")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
                
            frame = cv2.flip(frame, 1)
            
            face_results, hand_results = detector.process(frame)
            faces = detector.extract_faces(face_results, frame.shape)
            dets = [f['bbox'] for f in faces]
            
            tracked_faces = tracker.update(dets)
            
            # update callback params dynamically
            cv2.setMouseCallback('LectureLens', mouse_callback, {
                'state_manager': state_manager,
                'tracked_faces': tracked_faces
            })

            attributes_list = []
            for track in tracked_faces:
                x1, y1, x2, y2, track_id = map(int, track)
                track_center = ((x1 + x2) / 2, (y1 + y2) / 2)
                
                best_match = None
                min_dist = float('inf')
                for face in faces:
                    fx1, fy1, fx2, fy2, _ = face['bbox']
                    face_center = ((fx1 + fx2) / 2, (fy1 + fy2) / 2)
                    dist = math.hypot(track_center[0] - face_center[0], track_center[1] - face_center[1])
                    if dist < min_dist:
                        min_dist = dist
                        best_match = face
                
                if best_match and min_dist < 100:
                    landmarks = best_match['landmarks']
                    pose, _ = analyzer.get_head_pose(landmarks, frame.shape)
                    ear = analyzer.get_ear(landmarks, frame.shape)
                    emotion, _ = analyzer.get_emotion(frame, best_match['bbox'])
                    hand_raised = analyzer.check_hand_raised(best_match['bbox'], hand_results, frame.shape)
                    
                    # Check if new student before updating
                    if track_id not in state_manager.students:
                        if not os.path.exists("faces"):
                            os.makedirs("faces")
                        fx1, fy1, fx2, fy2 = map(int, best_match['bbox'][:4])
                        fh, fw, _ = frame.shape
                        fx1, fy1 = max(0, fx1), max(0, fy1)
                        fx2, fy2 = min(fw, fx2), min(fh, fy2)
                        face_img = frame[fy1:fy2, fx1:fx2]
                        if face_img.size > 0:
                            img_path = os.path.join("faces", f"student_{track_id}.jpg")
                            cv2.imwrite(img_path, face_img)
                            print(f"Saved photo for new student ID: {track_id} to {img_path}")
                    
                    color, score = state_manager.update_student(track_id, pose, ear, emotion, hand_raised)
                    
                    attributes_list.append({
                        'pose': pose,
                        'ear': ear,
                        'emotion': emotion,
                        'hand_raised': hand_raised,
                        'state_color': color
                    })
                else:
                    attributes_list.append({
                        'pose': 'Unknown',
                        'ear': 0.3,
                        'emotion': 'neutral',
                        'hand_raised': False,
                        'state_color': 'yellow'
                    })

            frame = ui.draw_bounding_boxes(frame, tracked_faces, attributes_list)
            dashboard_frame = ui.draw_dashboard(frame)

            cv2.imshow('LectureLens', dashboard_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('e'):
                print("Exporting report...")
                exporter.generate_pdf()
                print("Generating Gemini insights...")
                summary = gemini_analyzer.generate_summary(state_manager)
                print("\n=== AI SUMMARY ===\n")
                print(summary)
                print("\n==================\n")

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
