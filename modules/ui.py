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
        try:
            student = self.state_manager.students.get(student_id)
            if not student or not student.history or len(student.history) < 2:
                return np.zeros((400, self.dashboard_width, 3), dtype=np.uint8)

            now = time.time()
            if student_id in self.chart_cache and now - self.last_update_time.get(student_id, 0) < 1.0:
                return self.chart_cache[student_id]

            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg

            # Use Figure for thread safety (prevents the 'clicked face' crash)
            fig = Figure(figsize=(4, 4), dpi=100)
            canvas = FigureCanvasAgg(fig)
            fig.patch.set_facecolor('#1E1E1E')
            
            ax1, ax2 = fig.subplots(2, 1)
            ax1.set_facecolor('#1E1E1E')
            ax2.set_facecolor('#1E1E1E')
            
            recent_history = student.history
            times = [r['time'] for r in recent_history]
            scores = [r['score'] for r in recent_history]
            session_start = self.state_manager.session_start
            rel_times = [t - session_start for t in times]
            
            ax1.plot(rel_times, scores, color='cyan', linewidth=2)
            ax1.set_title(f"Student {student_id} Attention", color='white', fontsize=10)
            ax1.set_ylim(-10, 110)
            ax1.set_ylabel("Score", color='white', fontsize=8)
            ax1.tick_params(colors='white', labelsize=8)
            for spine in ax1.spines.values(): spine.set_visible(False)
            
            emotions = [record['emotion'] for record in student.history]
            unique_emotions = sorted(list(set(emotions)))
            counts = [emotions.count(e) for e in unique_emotions]
            
            if unique_emotions:
                ax2.bar(unique_emotions, counts, color='#FF57A0')
            ax2.set_title("Emotion Distribution", color='white', fontsize=10)
            ax2.tick_params(axis='x', rotation=45, colors='white', labelsize=8)
            ax2.tick_params(axis='y', colors='white', labelsize=8)
            for spine in ax2.spines.values(): spine.set_visible(False)
            
            fig.tight_layout()
            canvas.draw()
            rgba_buf = canvas.buffer_rgba()
            img = np.asarray(rgba_buf)
            
            bgr_img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR) if img.shape[2] == 4 else img[:,:,:3]
            
            if bgr_img.shape[1] != self.dashboard_width:
                bgr_img = cv2.resize(bgr_img, (self.dashboard_width, bgr_img.shape[0]))

            self.chart_cache[student_id] = bgr_img
            self.last_update_time[student_id] = now
            return bgr_img
        except Exception as e:
            return np.zeros((400, self.dashboard_width, 3), dtype=np.uint8)

    def draw_dashboard(self, frame):
        try:
            h, w, c = frame.shape
            dash = np.full((h, self.dashboard_width, 3), (30, 30, 30), dtype=np.uint8)
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            class_score = self.state_manager.calculate_class_engagement()
            score_color = self.colors['green'] if class_score > 60 else (self.colors['yellow'] if class_score > 30 else self.colors['red'])
            
            cv2.putText(dash, "LECTURE LENS", (20, 40), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            session_time = int(time.time() - self.state_manager.session_start)
            smins, ssecs = divmod(session_time, 60)
            cv2.putText(dash, f"{smins:02d}:{ssecs:02d}", (320, 40), font, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.line(dash, (20, 50), (380, 50), (80, 80, 80), 1)
            
            cv2.putText(dash, "Class Engagement", (20, 90), font, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(dash, f"{class_score}%", (20, 140), font, 1.5, score_color, 3, cv2.LINE_AA)
            
            focused_obj = self.state_manager.focused_student_id
            if focused_obj is not None:
                chart = self.generate_student_graph(focused_obj)
                ch, cw, _ = chart.shape
                
                # Rescale chart if the dashboard is too short (prevents broadcast crash)
                if h < 600:
                    new_ch = max(100, h - 200)
                    chart = cv2.resize(chart, (self.dashboard_width, new_ch))
                    ch = new_ch

                start_y = 200
                if start_y + ch <= h:
                    dash[start_y:start_y+ch, :self.dashboard_width] = chart
                else:
                    dash[h-ch:h, :self.dashboard_width] = chart

            cv2.putText(dash, "[Q] Quit  [E] Export Report", (20, h - 20), font, 0.5, (100, 100, 100), 1, cv2.LINE_AA)
            cv2.line(frame, (w-1, 0), (w-1, h), (50, 50, 50), 2)
            return np.hstack((frame, dash))
        except Exception:
            return frame

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
