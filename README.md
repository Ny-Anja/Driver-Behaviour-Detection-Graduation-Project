# Driver Behaviour Detection – Graduation Project

**Cairo University – Faculty of Computer Science and Artificial Intelligence**

A deep learning project for detecting and classifying distracted driver behaviour and seatbelt usage from dashboard camera images and video, developed as a graduation project. Multiple model architectures were explored and compared, including a custom CNN, ResNet50 (transfer learning), YOLOv5 (object detection), and a combined YOLOv5 + Keras seatbelt classifier.

---

## Problem Statement

Given 2D dashboard camera images, the system classifies driver behaviour to determine whether the driver is driving safely or engaged in a distracted activity (texting, talking on the phone, eating, etc.). A separate module detects whether the driver is wearing a seatbelt.

---

## Repository Structure

| Notebook | Description |
|---|---|
| `distracted-driver-detection.ipynb` | Custom CNN trained on the State Farm dataset — baseline model |
| `Copy_of_distracted-driver-detection.ipynb` | Extended version with additional experiments |
| `GPwith_Resnet.ipynb` | Transfer learning using **ResNet50** (PyTorch) on the State Farm dataset |
| `DriverBehaviourYolov5.ipynb` | Driver behaviour detection using **YOLOv5** trained via Roboflow |
| `Driver_behaviour_detection_with_yolov5.ipynb` | Full YOLOv5 inference pipeline for driver behaviour |
| `seat_built.ipynb` | Seatbelt detection using **YOLOv5** (detection) + **Keras model** (classification) on video |

---

## Dataset

- **State Farm Distracted Driver Detection** (Kaggle)
  - 102,150 images across 10 behaviour classes
  - Downloaded via Kaggle API

**Classes:**

| Label | Behaviour |
|---|---|
| c0 | Safe driving |
| c1 | Texting – right |
| c2 | Talking on the phone – right |
| c3 | Texting – left |
| c4 | Talking on the phone – left |
| c5 | Operating the radio |
| c6 | Drinking |
| c7 | Reaching behind |
| c8 | Hair and makeup |
| c9 | Talking to passenger |

- **Seatbelt dataset** — custom dataset from Roboflow with two classes: `Seatbelt Worn` / `No Seatbelt Worn`

---

## Models

### 1. Custom CNN
- 4 convolutional layers with max pooling, dropout, and 2 fully connected layers
- Xavier weight initialisation, ReLU activations, Softmax output
- Input: 64×64 (or 224×224) RGB images normalised to [−1, 1]

### 2. ResNet50 (Transfer Learning)
- Pretrained ResNet50 fine-tuned on the State Farm dataset using PyTorch
- `torchvision.models` with custom classification head

### 3. YOLOv5 (Driver Behaviour)
- YOLOv5s architecture trained on a Roboflow-hosted distracted driving dataset
- Custom YAML configuration for number of classes and anchor boxes

### 4. YOLOv5 + Keras Seatbelt Classifier
- YOLOv5 detects the driver bounding box in each video frame
- Cropped region passed to a Keras `.h5` classifier to determine seatbelt status
- Bounding boxes drawn in green (worn) or red (not worn) with confidence score

---

## Requirements

```
torch
torchvision
tensorflow
keras
ultralytics
opencv-python
roboflow
numpy
pandas
matplotlib
seaborn
scikit-learn
tqdm
```

All notebooks are designed to run on **Google Colab** with GPU acceleration (T4).

---

## Usage

1. Open any notebook in Google Colab
2. Mount Google Drive and upload required weights/data
3. For Kaggle datasets, upload your `kaggle.json` API key
4. For Roboflow datasets, use your Roboflow API key
5. Run cells sequentially
