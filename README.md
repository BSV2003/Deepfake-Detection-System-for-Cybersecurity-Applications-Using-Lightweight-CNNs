# Deepfake Detection System for Cybersecurity Applications Using Lightweight CNNs

This repository presents a deep learning–based framework for detecting
deepfake videos using lightweight Convolutional Neural Networks (CNNs)
combined with Long Short-Term Memory (LSTM) networks. The work focuses on
evaluating and comparing multiple CNN–LSTM architectures for deepfake
detection across multiple benchmark datasets.

---

## 📌 Project Overview

Deepfake technology uses deep learning techniques to manipulate videos,
making it difficult to distinguish between real and fake content. Such
manipulated media poses serious threats in cybersecurity, digital forensics,
social media misinformation, and identity fraud.

This project proposes a deepfake detection system that combines spatial
feature extraction using CNN architectures (MobileNet and MesoNet) with
temporal sequence learning using LSTM networks to identify manipulated videos.

---

## 📊 Datasets

The models are trained and evaluated on the following datasets:

* **FaceForensics++**
* **Celeb-DF**
* **DeepFake Detection Challenge (DFDC)**

**Input:** Video frames extracted from real and fake videos
**Output:** Binary classification (Real / Fake)

> **Note:** Dataset files are not included in this repository.
> Users must download the datasets separately and configure dataset paths before training.

---

## 🧠 Models Implemented

The following architectures are implemented and evaluated:

* **MobileNet + LSTM (Baseline)**
* **MobileNet + LSTM (Tuned)**
* **MesoNet + LSTM (Baseline)**
* **MesoNet + LSTM (Tuned)**

CNN models extract spatial features from video frames, while LSTM models
capture temporal inconsistencies across video frames, improving deepfake
detection performance.

---

## 📈 Experimental Results

| Dataset         | MobileNet + LSTM (Tuned) | MesoNet + LSTM (Tuned) |
| --------------- | ------------------------ | ---------------------- |
| Celeb-DF        | 94.35%                   | 70.38%                 |
| DFDC            | 96.96%                   | 78.61%                 |
| FaceForensics++ | 95.48%                   | 68.93%                 |

**Best Performing Model:** MobileNet + LSTM (Tuned)

The MobileNet + LSTM architecture demonstrated superior performance compared
to the MesoNet + LSTM architecture across all datasets.

---

## 📂 Repository Structure

```text
Deepfake-Detection-System/
│
├── preprocessing/
│   ├── combine_datasets.py
│   ├── video_data_generator.py
│
├── training/
│   ├── train_mesonet_lstm.py
│   ├── train_mesonet_lstm_tuned.py
│   ├── train_mobilenet_lstm.py
│   ├── train_mobilenet_lstm_tuned.py
│
├── evaluation/
│   └── evaluate_models.py
│
├── visualization/
│   └── pm_visuals.py
│
├── models/
│   ├── mesonet_lstm_baseline.keras
│   ├── mesonet_lstm_tuned.keras
│   ├── mobilenet_lstm_baseline.keras
│   ├── mobilenet_lstm_tuned.keras
│
├── results/
│   ├── accuracy_graphs/
│   ├── confusion_matrices/
│   ├── roc_curves/
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## ▶️ Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Train models:

```bash
python training/train_mobilenet_lstm.py
python training/train_mesonet_lstm.py
```

Train tuned models:

```bash
python training/train_mobilenet_lstm_tuned.py
python training/train_mesonet_lstm_tuned.py
```

Evaluate models:

```bash
python evaluation/evaluate_models.py
```

Generate performance visualizations:

```bash
python visualization/pm_visuals.py
```

---

## 🔐 Applications

* Cybersecurity
* Digital Forensics
* Fake News Detection
* Social Media Content Verification
* Identity Fraud Detection
* Video Authentication Systems

