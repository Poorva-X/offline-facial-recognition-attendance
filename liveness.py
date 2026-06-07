import cv2 as cv
import mediapipe as mp
import numpy as np

# Setup MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Open camera
cap = cv.VideoCapture(0)

# Blink counter
blink_count = 0
eye_open = True

print("Camera opened. Please blink to verify liveness.")

def get_eye_ratio(landmarks, eye_points, frame_w, frame_h):
    # Get eye landmark coordinates
    points = [(int(landmarks[p].x * frame_w), 
               int(landmarks[p].y * frame_h)) for p in eye_points]
    
    # Vertical distance
    vertical = np.linalg.norm(
        np.array(points[1]) - np.array(points[5]))
    
    # Horizontal distance
    horizontal = np.linalg.norm(
        np.array(points[0]) - np.array(points[3]))
    
    # Eye ratio
    ratio = vertical / horizontal
    return ratio

# Eye landmark points
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

while True:
    ret, frame = cap.read()
    frame_h, frame_w = frame.shape[:2]
    
    # Convert to RGB for MediaPipe
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Get eye ratios
        left_ratio = get_eye_ratio(landmarks, LEFT_EYE, frame_w, frame_h)
        right_ratio = get_eye_ratio(landmarks, RIGHT_EYE, frame_w, frame_h)
        avg_ratio = (left_ratio + right_ratio) / 2
        
        # Detect blink
        if avg_ratio < 0.25:
            if eye_open:
                blink_count += 1
                eye_open = False
                print(f"Blink detected! Total blinks: {blink_count}")
        else:
            eye_open = True
        
        # Show status
        if blink_count >= 2:
            cv.putText(frame, "LIVENESS VERIFIED ✅", (30, 50),
                      cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv.putText(frame, f"Please blink {2 - blink_count} more time(s)", (30, 50),
                      cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    cv.imshow("Liveness Detection", frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()