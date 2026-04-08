import os
import sys
import subprocess
import urllib.request

def ensure_environment():
    print("[LectureLens] Checking system setup...")
    
    # 1. Check if required libraries are missing
    try:
        import cv2
        import mediapipe
        import fer
        import filterpy
        import scipy
        import matplotlib
        import reportlab
        import google.genai
        import dotenv
        import numpy
        import pandas
    except ImportError as e:
        print(f"[LectureLens] Missing required package ({e.name}). Installing from requirements.txt...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            print("[LectureLens] All missing packages installed successfully!")
        except Exception as install_error:
            print(f"[LectureLens] Warning: Failed to auto-install dependencies: {install_error}")
            print("[LectureLens] You may need to run: pip install -r requirements.txt manually.")

    # 2. Check and Download Missing AI Models
    models = {
        "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
        "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    }
    
    for task_file, url in models.items():
        if not os.path.exists(task_file):
            print(f"[LectureLens] Model file '{task_file}' is missing. Downloading automatically (this might take a minute)...")
            try:
                urllib.request.urlretrieve(url, task_file)
                print(f"[LectureLens] Successfully downloaded {task_file}!")
            except Exception as e:
                print(f"[LectureLens] Error downloading {task_file}: {e}")
                print(f"[LectureLens] Please download it manually from: {url}")

    # 3. Check for the .env configuration file
    if not os.path.exists(".env"):
        print("[LectureLens] No '.env' file found. Generating a secure template...")
        with open(".env", "w") as f:
            f.write("GEMINI_API_KEY=your_api_key_here\n")
        print("\n" + "="*50)
        print(" ACTION REQUIRED:")
        print(" A default '.env' file has been created in your folder.")
        print(" Please open it and insert your actual Gemini API Key ")
        print(" otherwise the AI Insights feature will not work.")
        print("="*50 + "\n")
        
    print("[LectureLens] System checks complete! Launching Application...\n")

if __name__ == "__main__":
    ensure_environment()
