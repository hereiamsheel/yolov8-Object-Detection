# 🚀 YOLOv8 Object Detection Project (Python)

A simple and efficient **Object Detection system using YOLOv8** built with Python.  
Supports detection on **images, videos, and real-time webcam**.

---

## 📁 Project Structure

~~~ 
Object Detection YOLO/
│
├── Running YOLO/
│   ├── test1.jpg
│   ├── test1.mp4
│   ├── test2.mp4
│   ├── yolo-basics.py
│   └── yolo-webcam.py
│
├── Yolo-Weights/
│   └── yolov8n.pt
│
└── README.md
~~~

---

## ✨ Features

- 🔍 Object detection on images  
- 🎥 Object detection on videos  
- 📷 Real-time webcam detection  
- ⚡ Fast inference using YOLOv8  
- 🧠 Uses pre-trained lightweight model (`yolov8n.pt`)  

---

## 🛠️ Requirements

~~~bash
pip install ultralytics opencv-python cvzone
~~~

---

## ⚙️ Setup

1. Clone the repository:

~~~bash
git clone https://github.com/your-username/yolov8-Object-Detection.git
cd yolov8-Object-Detection
~~~

2. Ensure weights file exists:

~~~
Yolo-Weights/yolov8n.pt
~~~

---

## ▶️ Usage

### 🖼️ Image / Video Detection

~~~bash
python "Running YOLO/yolo-basics.py"
~~~

Edit the file path inside the script to use:
- `test1.jpg`
- `test1.mp4`
- `test2.mp4`

---

### 📷 Webcam Detection

~~~bash
python "Running YOLO/yolo-webcam.py"
~~~

---

## 📊 Output

- Bounding boxes  
- Class labels  
- Confidence scores  

---

## 🧠 Model

- **Model:** yolov8n.pt  
- **Type:** Nano (fast & lightweight)  
- **Framework:** Ultralytics YOLOv8  

---

## 🚀 Future Improvements

- 📊 Dashboard for analytics  
- 🌐 Web app (Flask / Streamlit)  
- 📦 Custom model training support  
- 💾 Export results (CSV/JSON)  

---

⭐ Star this repository if you found it useful!