# Offline Facial Recognition Attendance System

An AI-powered offline facial recognition and attendance 
system for field personnel in remote zero-network zones.

## Features
- Real-time face detection using OpenCV
- Face registration and recognition using DeepFace
- Liveness detection using MediaPipe (blink detection)
- Completely offline — works without internet
- Attendance logging with SQLite database
- AI Agent powered by Google Gemini
- AWS sync when internet is restored

## Tech Stack
- Python
- OpenCV
- DeepFace
- MediaPipe
- SQLite
- Google Gemini API
- FastAPI (backend)
- AWS RDS (cloud sync)

## How It Works
1. Supervisor registers each employee's face once
2. In the field — employee stands in front of camera
3. System asks employee to blink (liveness check)
4. Face is recognized and matched against database
5. Attendance is logged locally with name, ID, time
6. When internet returns — data syncs to AWS

## Installation
pip install opencv-python deepface mediapipe
pip install google-genai python-dotenv sqlite3
## Usage
Register an employee:
python detection_registeration.py

Mark attendance:
python attendance.py

## Project Structure
├── detection_registeration.py  # Face registration
├── recognition.py              # Face recognition
├── liveness.py                 # Blink detection
├── attendance.py               # Complete attendance flow
├── agent.py                    # Gemini AI agent
├── database.py                 # SQLite database setup
└── faces_database/             # Registered face photos

## Team
- AI/ML: Poorva (face recognition, liveness, AI agent)
- Backend: Shreya Sharma (FastAPI, AWS sync)