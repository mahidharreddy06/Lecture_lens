import cv2
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.style.use('dark_background')

class UI:
    def __init__(self, state_manager):
        self.state_manager = state_manager
        # Modern, softer color palette
        self.colors = {
            'green': (120, 240, 120),
            'yellow': (100, 210, 250), # Soft amber
            'red': (90, 90, 240)       # Soft red
        }
        self.dashboard_width = 400
        self.chart_cache = {}
        self.last_update_time = {}

    def draw_corner_rect(self, frame, x1, y1, x2, y2, color, thickness=2, length=15):
        # Top-left
        cv2.line(frame, (x1, y1), (x1 + length, y1), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x1, y1), (x1, y1 + length), color, thickness, cv2.LINE_AA)
        # Top-right
        cv2.line(frame, (x2, y1), (x2 - length, y1), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x2, y1), (x2, y1 + length), color, thickness, cv2.LINE_AA)
        # Bottom-left
        cv2.line(frame, (x1, y2), (x1 + length, y2), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x1, y2), (x1, y2 - length), color, thickness, cv2.LINE_AA)
        # Bottom-right
        cv2.line(frame, (x2, y2), (x2 - length, y2), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x2, y2), (x2, y2 - length), color, thickness, cv2.LINE_AA)

    def draw_text_bg(self, frame, text, x, y, bg_color=(40, 40, 40), text_color=(255, 255, 255)):
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        cv2.rectangle(frame, (x, y - th - 5), (x + tw + 6, y + baseline + 5), bg_color, -1)
        cv2.putText(frame, text, (x + 3, y), font, scale, text_color, thickness, cv2.LINE_AA)

    def draw_bounding_boxes(self, frame, tracked_faces, attributes_list):
        for i, track in enumerate(tracked_faces):
            x1, y1, x2, y2, track_id = map(int, track)
            attrs = attributes_list[i]
            color = self.colors.get(attrs['state_color'], (255, 255, 255))
            
            thickness = 3 if attrs['hand_raised'] else 2
            # Draw modern corner brackets instead of full box
            self.draw_corner_rect(frame, x1, y1, x2, y2, color, thickness=thickness, length=20)
            
            label = f"ID:{track_id} {attrs['emotion']} {attrs['pose']}"
            self.draw_text_bg(frame, label, x1, max(y1 - 15, 10), bg_color=(30, 30, 30), text_color=color)
            
            if attrs['hand_raised']:
                self.draw_text_bg(frame, "HAND RAISED", x1, y2 + 25, bg_color=(0, 150, 0), text_color=(255, 255, 255))
                
        return frame

    def generate_student_graph(self, student_id):
        student = self.state_manager.students.get(student_id)
        if not student or not student.history:
            return np.zeros((300, 400, 3), dtype=np.uint8)

        now = time.time()
        # Update graph once per second to prevent FPS drops
        if student_id in self.chart_cache and now - self.last_update_time.get(student_id, 0) < 1.0:
            return self.chart_cache[student_id]

        # Use dark background elements
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 4), dpi=100)
        fig.patch.set_facecolor('#1E1E1E')
        ax1.set_facecolor('#1E1E1E')
        ax2.set_facecolor('#1E1E1E')
        
        # Plot entire session history
        now_time = time.time()
        recent_history = student.history
        
        times = [r['time'] for r in recent_history]
        scores = [r['score'] for r in recent_history]
        
        # Plot relative to session start
        session_start = self.state_manager.session_start
        rel_times = [t - session_start for t in times]
        
        ax1.plot(rel_times, scores, color='cyan', linewidth=2)
        ax1.set_title(f"Student {student_id} Attention", color='white', pad=10)
        
        max_time = max(now_time - session_start, 10.0)
        ax1.set_xlim(0, max_time)
        ax1.set_ylabel("Score", color='white')
        ax1.set_ylim(-10, 110)
        
        # Remove spines for cleaner look
        for spine in ax1.spines.values():
            spine.set_visible(False)
            
        # Calculate emotion tallies
        emotions = [record['emotion'] for record in student.history]
        unique_emotions = list(set(emotions))
        counts = [emotions.count(e) for e in unique_emotions]
        
        ax2.bar(unique_emotions, counts, color='#FF57A0') # Magenta-ish accent
        ax2.set_title("Emotion Distribution", color='white', pad=10)
        ax2.tick_params(axis='x', rotation=45, colors='white')
        
        for spine in ax2.spines.values():
            spine.set_visible(False)
        
        fig.tight_layout()
        fig.canvas.draw()
        
        img = np.asarray(fig.canvas.buffer_rgba())
        plt.close(fig)
        
        bgr_img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        self.chart_cache[student_id] = bgr_img
        self.last_update_time[student_id] = now
        return bgr_img

    def draw_dashboard(self, frame):
        h, w, c = frame.shape
        # Create a dark sophisticated background instead of pure black
        dash = np.full((h, self.dashboard_width, 3), (30, 30, 30), dtype=np.uint8)
        
        # Modern fonts
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Draw class engagement prominently on the dashboard
        class_score = self.state_manager.calculate_class_engagement()
        score_color = self.colors['green'] if class_score > 60 else (self.colors['yellow'] if class_score > 30 else self.colors['red'])
        
        cv2.putText(dash, "LECTURE LENS", (20, 40), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        session_time = int(time.time() - self.state_manager.session_start)
        smins, ssecs = divmod(session_time, 60)
        cv2.putText(dash, f"{smins:02d}:{ssecs:02d}", (320, 40), font, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.line(dash, (20, 50), (380, 50), (80, 80, 80), 1)
        
        cv2.putText(dash, "Class Engagement", (20, 90), font, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(dash, f"{class_score}%", (20, 140), font, 1.5, score_color, 3, cv2.LINE_AA)

        cv2.putText(dash, "Click a face to inspect", (20, 180), font, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
        
        focused_obj = self.state_manager.focused_student_id
        if focused_obj is not None:
            student = self.state_manager.students.get(focused_obj)
            l_time = int(student.get_listened_time()) if student else 0
            lmins, lsecs = divmod(l_time, 60)
            cv2.putText(dash, f"Focus Time: {lmins:02d}:{lsecs:02d}", (220, 180), font, 0.5, self.colors['green'], 1, cv2.LINE_AA)
            
            chart = self.generate_student_graph(focused_obj)
            ch, cw, _ = chart.shape
            # Drop it in the dashboard below the overall stats
            start_y = 200
            if start_y + ch <= h:
                dash[start_y:start_y+ch, :self.dashboard_width] = chart
            else:
                dash[h-ch:h, :self.dashboard_width] = chart

        cv2.putText(dash, "[Q] Quit  [E] Export Report", (20, h - 20), font, 0.5, (100, 100, 100), 1, cv2.LINE_AA)

        # Draw a subtle separator line
        cv2.line(frame, (w-1, 0), (w-1, h), (50, 50, 50), 2)
        
        combined = np.hstack((frame, dash))
        return combined

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        state_manager = param['state_manager']
        tracked_faces = param['tracked_faces']
        
        for track in tracked_faces:
            x1, y1, x2, y2, track_id = map(int, track)
            if x1 <= x <= x2 and y1 <= y <= y2:
                state_manager.focused_student_id = track_id
                print(f"Focused on tracking ID {track_id}")
                break
