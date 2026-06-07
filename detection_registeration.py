import cv2 as cv
import os

# Load face detector
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open camera
cap = cv.VideoCapture(0)

print("Camera opened. Look at camera and press S to register face.")

while True:
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # Draw box around face
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv.putText(frame, "Press S to Register", (x, y-10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    cv.imshow("Register Employee", frame)
    
    key = cv.waitKey(1) & 0xFF
    
    # Press S to save
    if key == ord('s'):
        if len(faces) == 0:
            print("No face detected. Please look at camera.")
        else:
            name = input("Enter Employee Name: ")
            emp_id = input("Enter Employee ID: ")
            
            folder_name = f"faces_database/{name}_{emp_id}"
            os.makedirs(folder_name, exist_ok=True)
            
            photo_path = f"{folder_name}/photo.jpg"
            cv.imwrite(photo_path, frame)
            
            print(f"✅ {name} registered successfully!")
            print(f"Photo saved at: {photo_path}")
            break
    
    if key == ord('q'):
        break

cap.release()
cv.destroyAllWindows()