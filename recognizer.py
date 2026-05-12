import cv2
import csv
from datetime import datetime

# Names dictionary
names = {
    1: "Saroj"
}

# Attendance list
marked_names = []

# Attendance function
def mark_attendance(name):

    if name not in marked_names:

        with open("attendance.csv", "a", newline="") as file:

            writer = csv.writer(file)

            date_now = datetime.now().strftime("%d-%m-%Y")

            time_now = datetime.now().strftime("%H:%M:%S")

            writer.writerow([name, date_now, time_now])

            marked_names.append(name)

            print(f"{name} attendance marked")


# Load recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.read('trainer/trainer.yml')

# Load face detector
faceCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Start webcam
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Check camera
if not cam.isOpened():
    print("Camera not working")
    exit()

font = cv2.FONT_HERSHEY_SIMPLEX

while True:

    ret, img = cam.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(100, 100),
    )

    for (x, y, w, h) in faces:

        # Draw rectangle around face
        cv2.rectangle(
            img,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )

        # Predict face
        id, confidence = recognizer.predict(
            gray[y:y+h, x:x+w]
        )

        # If face recognized
        if confidence < 100:

            name = names[id]

            confidence_text = f"{round(100 - confidence)}% Match"

            # Mark attendance
            mark_attendance(name)

            # Show VERIFIED message
            cv2.putText(
                img,
                f"{name} VERIFIED",
                (x, y+h+30),
                font,
                0.8,
                (0, 255, 0),
                2
            )

        else:

            name = "Unknown"

            confidence_text = "0% Match"

            # Show UNKNOWN message
            cv2.putText(
                img,
                "UNKNOWN PERSON",
                (x, y+h+30),
                font,
                0.8,
                (0, 0, 255),
                2
            )

        # Show name
        cv2.putText(
            img,
            str(name),
            (x+5, y-5),
            font,
            1,
            (255, 255, 255),
            2
        )

        # Show confidence
        cv2.putText(
            img,
            str(confidence_text),
            (x+5, y+h-5),
            font,
            0.8,
            (255, 255, 0),
            2
        )

    # Show live webcam
    cv2.imshow(
        'Smart Attendance System',
        img
    )

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cam.release()
cv2.destroyAllWindows()