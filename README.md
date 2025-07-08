# -Hand-Gesture-Recognition-using-YOLOv11-

This project demonstrates how to detect **hand gesture** from real-time webcam using a custom-trained **YOLOv11** model. It leverages Ultralytics' YOLO for fast and accurate inference, making it ideal for real-time applications.

---

## 📓 Google Colab Notebook

[Open in Colab](https://colab.research.google.com/drive/1IkMewT7DTj9mS5oc7cF5ij0g3N-FpKM5?authuser=2)

---

## 🧠 Framework

- **Model**: YOLOv11 (You Only Look Once - Ultralytics)
- **Framework**: Python 3.10 using [Ultralytics YOLO](https://docs.ultralytics.com/)
- **Approach**:
  - Hand Gesture labels assigned to hand regions (e.g., okay,peace)
  - Trained YOLOv11 model for bounding box + class prediction
  - Inference supports webcam, video, or image files

---

## 🏁 How to Run Locally (After Training)

Follow the steps below to run the emotion detection system on your machine using the trained model (`best.pt`):

### ✅ Step 1: Create and activate a virtual environment in anaconda prompt

conda create -n facial python=3.10 -y
conda activate facial

### ✅ Step 2: Install Ultralytics YOLO

pip install ultralytics

### ✅ Step 3: Run Emotion Detection

Use the following command to run prediction:
After activating the the virtual environment make a test.py file with this code and run in your virtual environment

from ultralytics import YOLO
from PIL import Image
import cv2

model = YOLO("best.pt/your_model_name")
results = model.predict(source="0", show=True)  

## 🧪 Training the Model

To train the YOLOv11 model on your custom emotion-labeled dataset:

yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=50 imgsz=640

Make sure your data.yaml is formatted like this:
     
    train: /content/Common-Hand-Gestures-(Emoji)-6/train                                                   
    val: /content/Common-Hand-Gestures-(Emoji)-6/valid                                            
    test: /content/Common-Hand-Gestures-(Emoji)-6/test
    c: 24
    names: ['Call-Me', 'Fingers-Crossed', 'Fingers-Spread', 'Hand-Raised', 'Heart', 'Italian-Gesture', 'Left-                                                                    Fist', 'Live-Long-Prosper', 'Love-You', 'Middle', 'Okay', 'Oncoming-Punch', 'Peace', 'Pinched-                                                                      Fingers', 'Point-Down', 'Point-Left', 'Point-Right', 'Pointing-Up', 'Praying-Hands', 'Raised-                                                                       Fist', 'Right-Fist', 'Rock', 'Thumbs-Down', 'Thumbs-Up']
                                                          
    roboflow:
     workspace: eli-juergens-bbemu
     project: common-hand-gestures-emoji
     version: 6
     license: CC BY 4.0
     url: https://universe.roboflow.com/eli-juergens-bbemu/common-hand-gestures-emoji/dataset/6

---

## 📊 Performance

Model: YOLOv11m

Dataset: https://universe.roboflow.com/eli-juergens-bbemu/common-hand-gestures-emoji/dataset/6

Use Cases: Human-computer interaction, emotion-based response systems, assistive tech

