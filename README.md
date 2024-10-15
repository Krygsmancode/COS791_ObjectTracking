# COS791_ObjectTracking

Step 1: Set up your development environment (e.g., Python with OpenCV).
Step 2: Write initial code to load the provided videos and process each frame.
Step 3: Implement the detection algorithm to identify the ball.
Step 4: Develop the tracking and enhancement mechanisms.
Step 5: Test with provided video snippets and adjust for challenging conditions like low lighting or occlusions.
Step 6: Record a demonstration video showcasing your algorithm in action.
Step 7: Prepare a detailed report, including algorithm explanations, challenges faced, and evaluations.

object_tracking_project/
│
├── data/
│   ├── videos/
│   │   ├── input_video1.mp4
│   │   ├── input_video2.mp4
│   │   └── ... (other provided video files)
│   └── output/
│       ├── tracked_video1.mp4
│       ├── tracked_video2.mp4
│       └── ... (output videos after processing)
│
├── src/
│   ├── main.py
│   ├── ball_detection.py
│   ├── ball_tracking.py
│   ├── enhancement.py
│   └── utils.py
│
├── notebooks/
│   ├── exploration.ipynb
│   └── testing.ipynb
│
├── models/  # If you decide to use pre-trained models or save trained models
│   └── ... (saved model files, if any)
│
├── tests/
│   ├── test_ball_detection.py
│   ├── test_ball_tracking.py
│   └── test_enhancement.py
│
├── requirements.txt  # Python dependencies
│
├── README.md  # Project overview and instructions
│
├── report/
│   ├── report.docx  # Detailed report with sections
│   └── presentation.mp4  # Pre-recorded project presentation
│
└── config/
    ├── settings.yaml  # Configuration for different parameters
    └── logging.conf  # Logging configuration
