from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import matplotlib.pyplot as plt
import os
import shutil

class Exporter:
    def __init__(self, state_manager):
        self.state_manager = state_manager

    def generate_pdf(self, output_path="session_report.pdf"):
        if not os.path.exists('tmp_graphs'):
            os.makedirs('tmp_graphs')

        try:
            c = canvas.Canvas(output_path, pagesize=letter)
            width, height = letter

            c.setFont("Helvetica-Bold", 20)
            c.drawString(50, height - 50, "LectureLens Session Report")

            c.setFont("Helvetica", 14)
            c.drawString(50, height - 100, f"Total Students Tracked: {len(self.state_manager.students)}")
            
            avg_class_score = 0
            if self.state_manager.global_history:
                avg_class_score = int(sum([h['score'] for h in self.state_manager.global_history]) / len(self.state_manager.global_history))
            c.drawString(50, height - 120, f"Average Class Engagement: {avg_class_score}%")
            
            import time
            session_sec = int(time.time() - self.state_manager.session_start)
            smins, ssecs = divmod(session_sec, 60)
            c.drawString(50, height - 140, f"Session Duration: {smins}m {ssecs}s")

            y_pos = height - 160
            for s_id, student in self.state_manager.students.items():
                if y_pos < 200:
                    c.showPage()
                    y_pos = height - 50
                
                c.setFont("Helvetica-Bold", 12)
                c.drawString(50, y_pos, f"Student ID: {s_id}")
                
                 # create plot
                if student.history:
                    emotions = [r['emotion'] for r in student.history]
                    dominant_emo = max(set(emotions), key=emotions.count) if emotions else "unknown"
                    c.setFont("Helvetica", 10)
                    c.drawString(50, y_pos - 15, f"Dominant Emotion: {dominant_emo.capitalize()}")
                    
                    l_sec = int(student.get_listened_time())
                    lmins, lsecs = divmod(l_sec, 60)
                    c.drawString(50, y_pos - 30, f"Focus Time: {lmins}m {lsecs}s")

                    times = [r['time'] - self.state_manager.session_start for r in student.history]
                    scores = [r['score'] for r in student.history]
                    plt.figure(figsize=(4, 2))
                    plt.plot(times, scores)
                    plt.ylim(0, 110)
                    plt.title(f"Student {s_id} Engagement")
                    img_path = f"tmp_graphs/student_{s_id}.png"
                    plt.savefig(img_path)
                    plt.close()

                    c.drawImage(img_path, 50, y_pos - 150, width=300, height=140)
                y_pos -= 180

            c.save()
        finally:
            shutil.rmtree('tmp_graphs', ignore_errors=True)
            
        abs_path = os.path.abspath(output_path)
        print(f"Report exported to {abs_path}")
        return output_path
