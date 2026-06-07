import cv2 as cv
import mediapipe as mp
import numpy as np
from deepface import DeepFace
import os
import csv
from datetime import datetime

# Setup MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Load face detector
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open camera
cap = cv.VideoCapture(0)

# Blink tracking
blink_count = 0
eye_open = True
liveness_verified = False
attendance_marked = False

# Eye landmark points
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def get_eye_ratio(landmarks, eye_points, frame_w, frame_h):
    points = [(int(landmarks[p].x * frame_w),
               int(landmarks[p].y * frame_h)) for p in eye_points]
    vertical = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
    horizontal = np.linalg.norm(np.array(points[0]) - np.array(points[3]))
    ratio = vertical / horizontal
    return ratio

def save_attendance(name, emp_id):
    now = datetime.now()
    date = now.strftime("%d-%m-%Y")
    time = now.strftime("%H:%M:%S")
    
    # Save to CSV file
    with open("attendance_records.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, emp_id, date, time, "Present"])
    
    print(f"✅ Attendance marked for {name} | {emp_id} | {date} | {time}")

print("Camera opened. Please blink twice to verify liveness.")

while True:
    ret, frame = cap.read()
    frame_h, frame_w = frame.shape[:2]
    
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # Step 1 - Liveness Detection
    if not liveness_verified:
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            left_ratio = get_eye_ratio(landmarks, LEFT_EYE, frame_w, frame_h)
            right_ratio = get_eye_ratio(landmarks, RIGHT_EYE, frame_w, frame_h)
            avg_ratio = (left_ratio + right_ratio) / 2
            
            if avg_ratio < 0.25:
                if eye_open:
                    blink_count += 1
                    eye_open = False
            else:
                eye_open = True
            
            if blink_count >= 2:
                liveness_verified = True
                print("Liveness verified. Recognizing face...")
            else:
                cv.putText(frame, f"Blink {2 - blink_count} more time(s)", (30, 50),
                          cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Step 2 - Face Recognition
    elif liveness_verified and not attendance_marked:
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            face_crop = frame[y:y+h, x:x+w]
            
            try:
                result = DeepFace.find(
                    img_path=face_crop,
                    db_path="faces_database",
                    enforce_detection=False,
                    silent=True
                )
                
                if len(result) > 0 and len(result[0]) > 0:
                    match_path = result[0].iloc[0]['identity']
                    folder = os.path.basename(os.path.dirname(match_path))
                    name, emp_id = folder.rsplit('_', 1)
                    
                    label = f"{name} | {emp_id}"
                    color = (0, 255, 0)
                    
                    # Save attendance
                    save_attendance(name, emp_id)
                    attendance_marked = True
                    
                else:
                    label = "Unknown Person"
                    color = (0, 0, 255)
            
            except:
                label = "Recognizing..."
                color = (0, 255, 255)
            
            cv.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv.putText(frame, label, (x, y-10),
                      cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # Step 3 - Attendance Marked
    elif attendance_marked:
        cv.putText(frame, "ATTENDANCE MARKED SUCCESSFULLY", (30, 50),
                  cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv.putText(frame, "Press Q to exit", (30, 90),
                  cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    cv.imshow("Attendance System", frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()