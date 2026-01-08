#  DeepFake Detector API

> **Status:** Phase 2 Complete (Model + Backend API)  
> **Model:** ResNet-18 (High-Res Face Fine-Tuned)  
> **Current Capability:** API accepts images and returns Real/Fake probabilities.

## About The Project
This project is an AI system designed to detect AI-generated images (Deepfakes). 
Currently, the **Deep Learning Model** and **FastAPI Backend** are fully functional.

## Project Structure
```text
AI_Detector/
├── api/
│   ├── main.py                # FastAPI Server
│   └── resnet18_deepfake.pth  # Trained Model Weights
├── model.ipynb                # Training Pipeline (PyTorch)
└── requirements.txt           # Dependencies