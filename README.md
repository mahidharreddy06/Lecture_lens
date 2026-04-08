# Lecture Lens

Lecture Lens is an advanced, automated classroom monitoring and engagement analysis system. It leverages real-time computer vision and generative AI to track student attendance, monitor focus levels, analyze emotional states, and provide intelligent summaries of classroom sessions.

## Core Capabilities

*   **Real-time Student Tracking**: Utilizes state-of-the-art bounding box tracking (SORT) to persistently identify and monitor multiple students simultaneously, even with occlusions.
*   **Behavioral Analysis**: Computes head pose (yaw/pitch/roll) and Eye Aspect Ratio (EAR) using Mediapipe Face Landmarker to accurately determine focus and fatigue.
*   **Emotion Recognition**: Integrates FER (Facial Expression Recognition) to log real-time emotional states, helping educators understand class sentiment.
*   **Interaction Detection**: Detects raised hands via Mediapipe Hand Landmarker, tracking active student participation.
*   **Automated Reporting**: Automatically generates detailed PDF session reports, charting student engagement timelines and overall performance metrics.
*   **AI-Powered Insights**: Integrates with the Google GenAI SDK (Gemini) to evaluate the aggregated classroom data and provide educators with an intelligent, text-based diagnostic summary of the session.

## Installation

Lecture Lens features a fully automated bootstrapping sequence. You do not need to manually install dependencies or fetch machine learning models.

1.  Clone this repository or download the ZIP file.
    ```bash
    git clone https://github.com/mahidharreddy06/Lecture_lens.git
    cd Lecture_lens
    ```
2.  Execute the main application file:
    ```bash
    python main.py
    ```

On its first execution, the system will automatically:
*   Install required Python packages via `pip` (OpenCV, Mediapipe, Google GenAI, etc.).
*   Download necessary machine learning `.task` files directly from Google.
*   Generate a template `.env` configuration file in the core directory.

## Configuration

To enable the AI integration for end-of-session summaries:
1. Open the `.env` file that was generated during the first run.
2. Replace the placeholder text with your actual Google Gemini API Key:
   `GEMINI_API_KEY=your_actual_api_key_here`

## Usage Instructions

*   **Start the System**: Run `python main.py`. The application will initialize the webcam and begin processing the feed.
*   **Monitoring**: The graphical interface will display bounding boxes, unique student IDs, and a live dashboard of aggregated focus states.
*   **Export Report**: Press the `E` key on your keyboard to instantly generate the PDF session report and retrieve the Gemini AI summary in the terminal.
*   **Quit**: Press the `Q` key on your keyboard to terminate the monitoring session safely.

## Architecture

The system is designed with a highly modular architecture for scalability:
*   `main.py`: Application entry point and orchestrator.
*   `setup_env.py`: Environment bootstrapper for dependencies and models.
*   `/modules/detection.py`: Mediapipe model handling and bbox extraction.
*   `/modules/tracking.py`: Object persistence across frames using SORT.
*   `/modules/analysis.py`: Mathematical computation of pose, EAR, and emotions.
*   `/modules/state.py`: Centralized state management for student metrics.
*   `/modules/ui.py`: OpenCV drawing utilities and dashboard generation.
*   `/modules/export.py`: PDF generation logic via ReportLab.
*   `/modules/gemini.py`: Integration with Google GenAI for text summaries.
