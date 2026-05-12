import cv2
import os

# Create dataset folder
if not os.path.exists("dataset"):
    os.makedirs("dataset")

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

face_id = input("Enter ID: ")

count = 0

while True:

    ret, img = cam.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        cv2.rectangle(img, (x, y),
                      (x+w, y+h),
                      (255, 0, 0), 2)

        count += 1

        # Save image
        cv2.imwrite(
            f"dataset/User.{face_id}.{count}.jpg",
            gray[y:y+h, x:x+w]
        )

    cv2.imshow("Capturing Faces", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    elif count >= 50:
        break

cam.release()
cv2.destroyAllWindows()

print("Face samples collected")