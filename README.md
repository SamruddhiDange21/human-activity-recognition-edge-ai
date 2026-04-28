# human-activity-recognition-edge-ai
Real-time human activity recognition system using a custom dataset and MobileNetV2 with OpenCV for live webcam inference.
# Real-Time Human Activity Recognition (Edge AI)

## 📌 Overview

This project is a real-time human activity recognition system built using a custom dataset and a lightweight deep learning model. It uses a webcam feed to classify basic human activities such as **sitting** and **standing**.

The goal of this project was to understand the **end-to-end machine learning pipeline**, from data collection to real-time inference.

---

## 🎯 Problem Statement

Most beginner ML projects focus only on training models on clean datasets.
This project focuses on building a **complete system**:

* Capture real-world data
* Train a model on that data
* Deploy it for real-time predictions

---

## 🧠 Approach

### 1. Data Collection

* Created a custom dataset using webcam
* Classes:

  * Sitting
  * Standing
* Images captured manually using OpenCV
* Initial dataset had limited variation (same background)

---

### 2. Model

* Model used: **MobileNetV2 (lightweight CNN)**
* Trained from scratch (no pretrained weights)
* Modified final layer for 2-class classification

---

### 3. Training

* Loss function: CrossEntropyLoss
* Optimizer: Adam
* Trained for multiple epochs on custom dataset

---

### 4. Real-Time Inference

* Used OpenCV to capture webcam frames
* Each frame is passed through the trained model
* Predictions displayed live on screen

---

## ⚙️ Tech Stack

* Python
* PyTorch
* OpenCV
* Torchvision

---

## 🚀 Features

* **Continuous Data Collection**: Automated image capturing mode in `collect_data.py`.
* **Robust Training Pipeline**: Includes Train/Validation split and data augmentation in `train.py`.
* **Real-time Inference Enhancements**: Added real-time FPS tracking and confidence score displays in `predict_webcam.py`.
* Custom dataset creation pipeline
* Lightweight MobileNetV2 model suitable for edge environments
* End-to-end ML workflow implementation

---

## ⚠️ Limitations

* Model performs best in environments similar to training data
* Limited generalization due to:

  * Same background in initial dataset
  * Limited number of subjects
* Only 2 activity classes

---

## 🔧 Improvements (Next Steps)

* Add more activities (walking, waving)
* Improve dataset diversity (different people, backgrounds)
* Try other architectures to see how they perform on edge devices
* Deploy to an actual edge device like Raspberry Pi

---

## 🧠 Key Learnings

* Data quality and diversity matter more than model complexity
* Models can learn unintended patterns (like background bias)
* Real-time deployment introduces new challenges beyond training

---

## 📂 Project Structure

```
edge-ai-project/
│
├── data/                 # Custom dataset
├── model.pth            # Trained model
├── train.py             # Training script
├── collect_data.py      # Dataset collection
├── predict_webcam.py    # Real-time inference
└── README.md
```



## 💡 Conclusion

This project demonstrates a complete machine learning pipeline, from data collection to deployment. While simple, it highlights key real-world challenges such as data bias and generalization, which are critical in building practical ML systems.

---
