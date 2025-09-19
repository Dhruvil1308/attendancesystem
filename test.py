from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime


from win32com.client import Dispatch

def speak(str1):
    speak=Dispatch(("SAPI.SpVoice"))
    speak.Speak(str1)

video=cv2.VideoCapture(0)

# Use script directory to build absolute paths so relative runs work from any CWD
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
facedetect_path = os.path.join(BASE_DIR, 'data', 'haarcascade_frontalface_default.xml')
facedetect = cv2.CascadeClassifier(facedetect_path)

names_path = os.path.join(BASE_DIR, 'data', 'names.pkl')
faces_path = os.path.join(BASE_DIR, 'data', 'faces_data.pkl')

with open(names_path, 'rb') as w:
    LABELS = pickle.load(w)
with open(faces_path, 'rb') as f:
    FACES = pickle.load(f)

# Ensure types and lengths match: FACES should be (n_samples, n_features)
FACES = np.asarray(FACES)
LABELS = list(LABELS)

print('Shape of Faces matrix --> ', FACES.shape)

# Fix common mismatch: if there are more labels than face samples, trim labels.
if len(LABELS) != FACES.shape[0]:
    if len(LABELS) > FACES.shape[0]:
        print(f"Warning: more labels ({len(LABELS)}) than face samples ({FACES.shape[0]}). Trimming labels to match faces.")
        LABELS = LABELS[: FACES.shape[0]]
    else:
        raise ValueError(f"Found fewer labels ({len(LABELS)}) than face samples ({FACES.shape[0]}). Please check your data files or re-run `add_faces.py` to add matching entries.")

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Load background image from project folder; fall back to a blank image if missing
bg_path = os.path.join(BASE_DIR, 'background.png')
imgBackground = cv2.imread(bg_path)
if imgBackground is None:
    print(f"Warning: background image not found at {bg_path}. Using blank background.")
    imgBackground = np.zeros((720, 1280, 3), dtype=np.uint8)

COL_NAMES = ['NAME', 'TIME']

while True:
    ret,frame=video.read()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=facedetect.detectMultiScale(gray, 1.3 ,5)
    for (x,y,w,h) in faces:
        crop_img=frame[y:y+h, x:x+w, :]
        resized_img=cv2.resize(crop_img, (50,50)).flatten().reshape(1,-1)
        output=knn.predict(resized_img)
        ts=time.time()
        date=datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp=datetime.fromtimestamp(ts).strftime("%H:%M-%S")

        attendance_dir = os.path.join(BASE_DIR, 'Attendance')
        os.makedirs(attendance_dir, exist_ok=True)
        attendance_file = os.path.join(attendance_dir, "Attendance_" + date + ".csv")
        exist = os.path.isfile(attendance_file)

        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
        cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
        cv2.putText(frame, str(output[0]), (x,y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)
        attendance=[str(output[0]), str(timestamp)]
    imgBackground[162:162 + 480, 55:55 + 640] = frame
    cv2.imshow("Frame",imgBackground)
    k=cv2.waitKey(1)
    if k==ord('o'):
        speak("Attendance Taken..")
        time.sleep(5)
        if exist:
            with open(attendance_file, "a", newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(attendance)
        else:
            with open(attendance_file, "a", newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(COL_NAMES)
                writer.writerow(attendance)
    if k==ord('q'):
        break
video.release()
cv2.destroyAllWindows()

