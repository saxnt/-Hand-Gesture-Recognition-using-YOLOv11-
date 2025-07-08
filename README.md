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
  - Emotion labels assigned to facial regions (e.g., happy, sad, angry)
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

    train: /content/Custom-Workflow-Object-Detection-11/train
    val: /content/Custom-Workflow-Object-Detection-11/valid
    test: /content/Custom-Workflow-Object-Detection-11/test
    nc: 7
    names: ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
    roboflow:
      workspace: hipster-pi5hv
      project: custom-workflow-object-detection-kuaeb
      version: 11
      license: CC BY 4.0
      url: https://universe.roboflow.com/hipster-pi5hv/custom-workflow-object-detection-kuaeb/dataset/11

---

## 📊 Performance

Model: YOLOv11m

Dataset: https://universe.roboflow.com/eli-juergens-bbemu/common-hand-gestures-emoji/dataset/6

Use Cases: Human-computer interaction, emotion-based response systems, assistive tech

