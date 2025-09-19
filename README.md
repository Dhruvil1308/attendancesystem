# 🎯 Face Recognition Attendance Project

This repository contains a simple **face-recognition based attendance system** implemented in Python using **OpenCV** and **scikit-learn**.  
It captures face images, stores face vectors and names, and uses a **K-Nearest Neighbors (KNN)** classifier to identify faces in a live camera feed and log attendance into CSV files.

---

## ✨ Features
- 📸 Add face data for a person using the webcam (`add_faces.py`).
- 💾 Store captured face vectors and names in the `data/` folder as pickle files.
- 🎥 Recognize faces in a live webcam stream and log attendance to dated CSV files (`test.py`).
- 🤖 Simple **KNN-based classifier** (`sklearn.neighbors.KNeighborsClassifier`).

---

## 📂 Repository Layout
```
.
├── add_faces.py                  # Capture face images for one person
├── test.py                       # Recognition & attendance script
├── app.py                        # (Optional) UI entrypoint
├── background.png                 # Optional background image
├── data/                          # Persistent storage
│   ├── faces_data.pkl             # Saved face vectors
│   ├── names.pkl                  # Names corresponding to vectors
│   └── haarcascade_frontalface_default.xml   # OpenCV face detector
├── Attendance/                    # Generated attendance CSVs
```

---

## ⚙️ Requirements
- Python **3.8+** (tested on 3.10/3.11)  
- OpenCV (`opencv-python`)  
- numpy  
- scikit-learn  
- pywin32 *(Windows only, for TTS `win32com.client.Dispatch`)*  

### 🔧 Installation
```powershell
# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install opencv-python numpy scikit-learn pywin32
```

👉 For full OpenCV contrib features:
```powershell
pip install opencv-contrib-python
```

---

## 🚀 Usage

### 1. Prepare data directory
Ensure `data/haarcascade_frontalface_default.xml` exists.  
📥 Download from [OpenCV repository](https://github.com/opencv/opencv/tree/master/data/haarcascades) if missing.

---

### 2. Add a person’s face samples
```powershell
& "D:/Python new/python.exe" "d:/SIH 2025/prototype/face_recognition_project/add_faces.py"
```
- Enter the person’s name when prompted.  
- Captures **100 cropped faces (50×50 px)**.  
- Stores samples in `faces_data.pkl` & `names.pkl`.

---

### 3. Run recognition & attendance
```powershell
& "D:/Python new/python.exe" "d:/SIH 2025/prototype/face_recognition_project/test.py"
```
- Opens webcam feed and shows detected faces with names.  
- Press **`o`** → take attendance → saves to `Attendance/Attendance_DD-MM-YYYY.csv`.  
- Press **`q`** → quit program.  

---

## 🛠️ Important Implementation Details
- Paths resolved relative to script directory → prevents working dir issues.  
- KNN model expects `faces_data.pkl` and `names.pkl` lengths to match.  
  - If mismatch: extra labels trimmed with warning / fewer labels → error.  
- Faces resized to `50x50` and flattened before training.  
- Attendance CSV format → 2 columns: `NAME`, `TIME`.  

---

## 🐞 Troubleshooting

- **Error:** `"Can't open file: 'data/haarcascade_frontalface_default.xml'"`  
  → Place cascade file inside `data/`.

- **Error:** `ValueError about inconsistent number of samples (e.g., [100, 300])`  
  → Mismatch between `faces_data.pkl` and `names.pkl`. Check with:
  ```python
  import pickle, os
  base = r"d:/SIH 2025/prototype/face_recognition_project"
  with open(os.path.join(base, 'data', 'faces_data.pkl'), 'rb') as f:
      faces = pickle.load(f)
  with open(os.path.join(base, 'data', 'names.pkl'), 'rb') as f:
      names = pickle.load(f)
  print('faces:', getattr(faces, 'shape', len(faces)))
  print('names:', len(names))
  ```
  → Re-run `add_faces.py` if mismatch persists.

- **Error:** `background.png not found`  
  → Add `background.png` in root, or script falls back to blank.

---

## 🚧 Next Improvements
- Save/load trained classifier (`joblib`/`pickle`) instead of retraining each run.  
- Use robust embedding models (e.g., FaceNet, dlib, face_recognition).  
- Add GUI (Tkinter/Flask) for user & attendance management.  
- Add unit tests and validation scripts for `data/`.  

---

## 🤝 Contributing
Contributions are welcome!  
- Open an issue for discussion.  
- Submit PRs with clear commit messages.  
- Keep backward compatibility for data format changes.
Licensed under the **MIT License**.  
You are free to use, modify, and distribute with attribution.  
