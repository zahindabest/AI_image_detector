# 🕵️‍♂️ AI Image Detector (DeepFake Detection)

> **Status:** Phase 1 Complete (Model Training & Evaluation)  
> **Accuracy:** 88.2% | **ROC-AUC:** 0.96 | **Recall (Fakes):** 93%

## 📖 About The Project
This project is a Deep Learning solution designed to distinguish between **Real** images and **AI-Generated** (Synthetic) images. With the rise of Generative AI (Midjourney, Stable Diffusion), distinguishing reality from fabrication is becoming a critical security challenge.

This repository contains the **training pipeline** and the **trained ResNet-18 model** fine-tuned on the **CIFAKE** dataset (60,000 Real / 60,000 Fake images).

## 📊 Performance Metrics
The model was evaluated on a test set of 20,000 images (10k Real, 10k Fake).

| Metric | Score | Interpretation |
| :--- | :--- | :--- |
| **ROC-AUC** | **0.9563** | Excellent ability to separate Real vs Fake classes. |
| **Recall (Fake)** | **0.93** | Captures 93% of all Deepfakes (High Security). |
| **Precision (Fake)**| **0.85** | When it claims an image is fake, it is correct 85% of the time. |
| **Accuracy** | **88.18%** | Overall correctness on the balanced test set. |

*Note: The model prioritizes Recall (catching fakes) over Precision, meaning it occasionally flags low-quality real images as fake to ensure no deepfakes slip through.*

## 🛠️ Tech Stack
* **Core:** Python 3.10+
* **DL Framework:** PyTorch (Torchvision)
* **Architecture:** ResNet-18 (Pre-trained on ImageNet)
* **Evaluation:** Scikit-Learn, Matplotlib, Seaborn

## 📂 Project Structure
```text
AI_Detector/
├── model.ipynb             # The full training & evaluation pipeline (Jupyter Notebook)
├── resnet18_deepfake.pth   # The saved model weights (PyTorch state_dictionary)
└── .gitignore              # Configurations to ignore the dataset folder