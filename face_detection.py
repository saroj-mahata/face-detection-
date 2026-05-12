import cv2
import csv
from datetime import datetime

# Store marked names
marked_names = []

# Attendance function
def mark_attendance(name):

    if name not in marked_names:

        with open("attendance.csv", "a", newline="") as file:

            writer = csv.writer(file)

            time_now = datetime.now().strftime("%H:%M:%S")

            writer.writerow([name, time_now])

            marked_names.append(name)

            print(f"{name} attendance marked")


# Load face detector
face_cap = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Open webcam
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:

    ret, frame = cam.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cap.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:

        # Draw rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (255, 0, 0),
            2
        )

        # Name display
        name = "Saroj"

        cv2.putText(
            frame,
            name,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        # Mark attendance
        mark_attendance(name)

    cv2.imshow("Live Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()