import os
from google import genai
from dotenv import load_dotenv

class AIAnalyzer:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key and api_key != "your_api_key_here":
            self.client = genai.Client(api_key=api_key)
            self.enabled = True
        else:
            self.enabled = False
            print("Gemini API key not found. AI insights disabled.")

    def generate_summary(self, state_manager):
        if not self.enabled:
            return "Gemini AI is disabled. Please provide an API key in .env"

        data = f"Total students tracked: {len(state_manager.students)}\n"
        
        overall = state_manager.global_history
        if overall:
            data += "Class Engagement History:\n"
            data += ", ".join([f"{int(r['time'] - state_manager.session_start)}s: {r['score']}%" for r in overall[::5]])
            
        data += "\nPer student summary:\n"
        for s_id, student in state_manager.students.items():
            if not student.history: continue
            avg = sum([r['score'] for r in student.history]) / len(student.history)
            emotions = [r['emotion'] for r in student.history]
            if emotions:
                common_emotion = max(set(emotions), key=emotions.count)
            else:
                common_emotion = "unknown"
            data += f"- ID {s_id}: Avg Attention {int(avg)}%, mostly {common_emotion}\n"

        prompt = (
            "You are an expert educational AI assistant analyzing data from a classroom session monitoring system. "
            "Write a concise, professional paragraph summarizing the session's engagement. "
            "Point out when engagement dropped, the overall mood, any students that were consistently confused or disengaged, "
            "and end with an actionable insight for the teacher.\n\n"
            f"Data:\n{data}"
        )
        
        try:
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            return response.text
        except Exception as e:
            return f"Error generating insights: {e}"
