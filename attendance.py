import csv
from datetime import datetime

def mark_attendance(name):

    with open("attendance.csv", "a", newline="") as file:

        writer = csv.writer(file)

        time_now = datetime.now().strftime("%H:%M:%S")

        writer.writerow([name, time_now])

        print(f"{name} attendance marked at {time_now}")

# Test
mark_attendance("Saroj")