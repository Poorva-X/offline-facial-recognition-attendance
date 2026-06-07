import cv2 as cv
from deepface import DeepFace
import os

# Load face detector
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open camera
cap = cv.VideoCapture(0)

print("Camera opened. Recognizing faces...")

while True:
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Crop just the face from frame
        face_crop = frame[y:y+h, x:x+w]

        try:
            # Ask DeepFace to find who this face is
            result = DeepFace.find(
                img_path=face_crop,
                db_path="faces_database",
                enforce_detection=False,
                silent=True
            )

            # If match found
            if len(result) > 0 and len(result[0]) > 0:
                # Get the matched file path
                match_path = result[0].iloc[0]['identity']
                
                # Extract name and ID from folder name
                folder = os.path.basename(os.path.dirname(match_path))
                name, emp_id = folder.rsplit('_', 1)
                
                label = f"{name} | {emp_id}"
                color = (0, 255, 0)  # Green for known person
            else:
                label = "Unknown"
                color = (0, 0, 255)  # Red for unknown

        except Exception as e:
            label = "Unknown"
            color = (0, 0, 255)

        # Draw box and name
        cv.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv.putText(frame, label, (x, y-10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv.imshow("Face Recognition", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()