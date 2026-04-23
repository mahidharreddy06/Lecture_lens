import setup_env
setup_env.ensure_environment()

import cv2
import math
import os
import numpy as np
import threading
import queue
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
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 60)
    
    cv2.namedWindow('LectureLens')
    
    mouse_data = {
        'state_manager': state_manager,
        'tracked_faces': []
    }
    cv2.setMouseCallback('LectureLens', mouse_callback, mouse_data)

    print("LectureLens started. Press 'Q' to quit, 'E' to export report.")

    saved_photos = set()
    emotion_histories = {}
    
    display_id_map = {}
    next_display_id_arr = [1]

    # Pre-populate saved photos if faces directory exists
    if os.path.exists("faces"):
        for f in os.listdir("faces"):
            if f.startswith("student_") and f.endswith(".jpg"):
                try: saved_photos.add(int(f.split('_')[1].split('.')[0]))
                except: pass

    # --- Async AI Threading Setup ---
    frame_queue = queue.Queue(maxsize=1)
    result_queue = queue.Queue(maxsize=1)

    def ai_worker():
        while True:
            frame = frame_queue.get()
            if frame is None:
                break  # Exit signal
            
            face_results, hand_results, object_results, fd_results = detector.process(frame)
            faces = detector.extract_faces(face_results, frame.shape)
            faces = detector.extract_short_range_faces(fd_results, faces)
            persons = detector.extract_persons(object_results)
            
            dets = [p['bbox'] for p in persons]
            
            # EfficientDet occasionally misses large close-up people. 
            # If a face is found but isn't inside any person bbox, create a surrogate body box.
            for face in faces:
                fx1, fy1, fx2, fy2, f_score = face['bbox']
                fcx, fcy = (fx1+fx2)/2, (fy1+fy2)/2
                
                covered = False
                for p in persons:
                    px1, py1, px2, py2, _ = p['bbox']
                    if px1 <= fcx <= px2 and py1 <= fcy <= py2:
                        covered = True
                        break
                
                if not covered:
                    w = fx2 - fx1
                    h = fy2 - fy1
                    # Shrink the surrogate fallback to a localized head/shoulders box
                    # This prevents foreground macro-faces from rendering full-screen bounding boxes
                    px1 = max(0, int(fx1 - w * 0.2))
                    py1 = max(0, int(fy1 - h * 0.2))
                    px2 = min(frame.shape[1], int(fx2 + w * 0.2))
                    py2 = min(frame.shape[0], int(fy2 + h * 0.8))
                    dets.append([px1, py1, px2, py2, f_score])
            
            ai_tracked_faces = tracker.update(dets)
            
            attributes_list = []
            for track in ai_tracked_faces:
                x1, y1, x2, y2, raw_id = map(int, track)
                
                # Force sequentially increasing UI display IDs (1, 2, 3...) 
                if raw_id not in display_id_map:
                    display_id_map[raw_id] = next_display_id_arr[0]
                    next_display_id_arr[0] += 1
                display_id = display_id_map[raw_id]
                
                best_match = None
                for face in faces:
                    fx1, fy1, fx2, fy2, _ = face['bbox']
                    fcx, fcy = (fx1+fx2)/2, (fy1+fy2)/2
                    if x1 <= fcx <= x2 and y1 <= fcy <= y2:
                        best_match = face
                        break
                
                if best_match:
                    landmarks = best_match['landmarks']
                    if landmarks is not None:
                        pose, _ = analyzer.get_head_pose(landmarks, frame.shape)
                        ear = analyzer.get_ear(landmarks, frame.shape)
                    else:
                        pose = "Forward"  # Default assumption for short-range detector
                        ear = 0.3
                    
                    emotion, _ = analyzer.get_emotion(frame, best_match['bbox'])
                    
                    if display_id not in emotion_histories:
                        emotion_histories[display_id] = []
                    emotion_histories[display_id].append(emotion)
                    if len(emotion_histories[display_id]) > 5:
                        emotion_histories[display_id].pop(0)
                        
                    stable_emotion = max(set(emotion_histories[display_id]), key=emotion_histories[display_id].count)
                    
                    hand_raised = analyzer.check_hand_raised(best_match['bbox'], hand_results, frame.shape)
                    
                    if display_id not in saved_photos and pose == 'Forward':
                        if not os.path.exists("faces"):
                            os.makedirs("faces")
                        fx_save, fy_save, fx2_save, fy2_save = map(int, best_match['bbox'][:4])
                        fh, fw, _ = frame.shape
                        fx_save, fy_save = max(0, fx_save), max(0, fy_save)
                        fx2_save, fy2_save = min(fw, fx2_save), min(fh, fy2_save)
                        face_img = frame[fy_save:fy2_save, fx_save:fx2_save]
                        if face_img.size > 0:
                            img_path = os.path.join("faces", f"student_{display_id}.jpg")
                            cv2.imwrite(img_path, face_img)
                            print(f"Saved clear photo for student ID: {display_id} to {img_path}")
                            saved_photos.add(display_id)
                    
                    color, score = state_manager.update_student(display_id, pose, ear, stable_emotion, hand_raised)
                    
                    attrs = {
                        'display_id': display_id,
                        'pose': pose, 'ear': ear, 'emotion': stable_emotion,
                        'hand_raised': hand_raised, 'state_color': color,
                        'ui_box': [fx1, fy1, fx2, fy2]
                    }
                else:
                    attrs = {
                        'display_id': display_id,
                        'pose': 'Unknown', 'ear': 0.3, 'emotion': 'neutral',
                        'hand_raised': False, 'state_color': 'yellow',
                        'ui_box': [x1, y1, x2, y2]
                    }
                
                attributes_list.append(attrs)

            # Prevent queue blocking
            try: result_queue.get_nowait()
            except queue.Empty: pass
            
            result_queue.put({
                'tracked_faces': ai_tracked_faces,
                'attributes_list': attributes_list
            })

    ai_thread = threading.Thread(target=ai_worker, daemon=True)
    ai_thread.start()

    tracked_faces = []
    attributes_list = []

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
                
            frame = cv2.flip(frame, 1)

            if not frame_queue.full():
                try: frame_queue.get_nowait()
                except queue.Empty: pass
                frame_queue.put(frame.copy())
            
            while not result_queue.empty():
                res = result_queue.get()
                tracked_faces = res['tracked_faces']
                attributes_list = res['attributes_list']
            
            # Create a scaled down version for UI to restore classic window size/stats proportions
            scale = 0.75
            display_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
            
            scaled_tracked_faces = []
            for i, track in enumerate(tracked_faces):
                x1, y1, x2, y2, _ = track
                # Override the returned tracking ID with the cleanly mapped sequential UI display ID
                display_id = attributes_list[i]['display_id']
                # Use the ui_box (tight face bounds) instead of Kalman body bounds for rendering
                ui_x1, ui_y1, ui_x2, ui_y2 = attributes_list[i]['ui_box']
                scaled_tracked_faces.append([ui_x1*scale, ui_y1*scale, ui_x2*scale, ui_y2*scale, display_id])
            
            # update callback params dynamically (using the scaled coords)
            mouse_data['tracked_faces'] = scaled_tracked_faces

            display_frame = ui.draw_bounding_boxes(display_frame, scaled_tracked_faces, attributes_list)
            dashboard_frame = ui.draw_dashboard(display_frame)

            cv2.imshow('LectureLens', dashboard_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('e'):
                print("Exporting report...")
                import time
                timestamp = int(time.time())
                report_name = f"session_report_{timestamp}.pdf"
                exporter.generate_pdf(report_name)
                print("Generating Gemini insights...")
                summary = gemini_analyzer.generate_summary(state_manager)
                print("\n=== AI SUMMARY ===\n")
                print(summary)
                print("\n==================\n")

    finally:
        frame_queue.put(None) # Signal AI thread to exit
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
