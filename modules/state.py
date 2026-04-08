import time

class StudentState:
    def __init__(self, student_id):
        self.student_id = student_id
        self.history = [] 
        
    def add_record(self, state_color, emotion, attention_score):
        self.history.append({
            'time': time.time(),
            'state': state_color,
            'emotion': emotion,
            'score': attention_score
        })

    def get_listened_time(self):
        if len(self.history) < 2: return 0
        total_time = 0
        for i in range(1, len(self.history)):
            diff = self.history[i]['time'] - self.history[i-1]['time']
            # Count if continuously tracked (gap < 2s) and 'attentive' (score >= 50)
            if diff < 2.0 and self.history[i]['score'] >= 50:
                total_time += diff
        return total_time

class StateManager:
    def __init__(self):
        self.students = {}
        self.session_start = time.time()
        self.focused_student_id = None
        self.global_history = []

    def compute_engagement(self, pose, ear, emotion, hand_raised):
        if ear < 0.18:  # strictly drowsy
            return 'red', 20

        if pose in ['Left', 'Right']:
            return 'red', 30
            
        if pose == 'Down':
            return 'red', 20

        if pose in ['Forward', 'Up']:
            # If they are generally looking forward/at the board, they are paying attention.
            # Emotion no longer governs their attention score to be negative.
            return 'green', 100
            
        return 'yellow', 50

    def update_student(self, track_id, pose, ear, emotion, hand_raised):
        if track_id not in self.students:
            self.students[track_id] = StudentState(track_id)
            
        color, score = self.compute_engagement(pose, ear, emotion, hand_raised)
        
        # Boost if hand raised
        if hand_raised:
            color = 'green'
            score = 100
            
        self.students[track_id].add_record(color, emotion, score)
        return color, score
        
    def calculate_class_engagement(self):
        if not self.students:
            return 100
        total_score = 0
        n_active = 0
        now = time.time()
        for s_id, student in self.students.items():
            # Only consider students seen in the last 5 seconds
            if student.history and now - student.history[-1]['time'] < 5:
                total_score += student.history[-1]['score']
                n_active += 1
        
        if n_active == 0:
            return 100
            
        class_score = int(total_score / n_active)
        self.global_history.append({'time': now, 'score': class_score})
        return class_score
