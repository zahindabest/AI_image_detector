from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import ViTForImageClassification, ViTImageProcessor
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import cv2
import numpy as np
import torch.nn.functional as F

app = FastAPI(title="Adaptive DeepFake Detector (Hybrid)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. LOAD MODEL A (The "Old School" ResNet) ---
# Good for: Blurry images, Bad Anatomy, Low-Res
def load_resnet():
    print("Loading Model A (ResNet)...")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    try:
        model.load_state_dict(torch.load("resnet18_deepfake.pth", map_location=device))
        print("Model A Loaded.")
    except:
        print("Model A weights missing. It will be random!")
    model = model.to(device)
    model.eval()
    return model

model_a = load_resnet()

# ResNet Preprocessing
transform_a = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 2. LOAD MODEL B (The "High Tech" ViT) ---
# Good for: Sharp images, Texture Analysis, Midjourney/Sora
MODEL_B_NAME = "dima806/deepfake_vs_real_image_detection"
def load_vit():
    print("Loading Model B (ViT)...")
    processor = ViTImageProcessor.from_pretrained(MODEL_B_NAME)
    model = ViTForImageClassification.from_pretrained(MODEL_B_NAME).to(device)
    print("Model B Loaded.")
    return processor, model

processor_b, model_b = load_vit()

# --- 3. HELPER: SHARPNESS DETECTOR ---
def get_sharpness_score(image_cv):
    # Convert to gray
    gray = cv2.cvtColor(image_cv, cv2.COLOR_RGB2GRAY)
    # Calculate Variance of Laplacian (Standard method for blur detection)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score

# Face Detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg", "image/webp"]:
        raise HTTPException(status_code=400, detail="Invalid file type.")

    try:
        # Read Image
        image_bytes = await file.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        cv_image = np.array(pil_image)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)

        # Detect Face
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        final_image = pil_image
        # Default sharpness (full image if no face)
        sharpness_image = cv_image 
        
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
            
            # --- FIX: TIGHT CROP FOR SHARPNESS CHECK ---
            # We crop EXACTLY the face (no padding) to ignore background blur
            sharpness_image = cv_image[y:y+h, x:x+w]
            
            # --- STANDARD CROP FOR MODEL (Keep Padding) ---
            p = 20
            x, y = max(0, x - p), max(0, y - p)
            w, h = w + (p * 2), h + (p * 2)
            final_image = pil_image.crop((x, y, x+w, y+h))

        # --- THE ROUTER LOGIC ---
        # Calculate score ONLY on the tight face crop
        sharpness = get_sharpness_score(sharpness_image)
        print(f"Face-Only Sharpness Score: {sharpness:.2f}")

        # Use 40 to catch darker textures/ curly hair
        SHARPNESS_THRESHOLD = 40 

        if sharpness < SHARPNESS_THRESHOLD:
            # ---> USE MODEL A (ResNet)
            print("Low Detail. Routing to Model A (ResNet).")
            used_model = "Model A (ResNet-18)"
            
            input_tensor = transform_a(final_image).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model_a(input_tensor)
                probs = F.softmax(outputs, dim=1)
                
            fake_prob = probs[0][0].item()
            real_prob = probs[0][1].item()
            
            if fake_prob > real_prob:
                label = "FAKE"
                conf = fake_prob
            else:
                label = "REAL"
                conf = real_prob

        else:
            # ---> USE MODEL B (ViT)
            print("High Detail. Routing to Model B (ViT).")
            used_model = "Model B (Vision Transformer)"
            
            inputs = processor_b(images=final_image, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model_b(**inputs)
                probs = F.softmax(outputs.logits, dim=1)
            
            idx = probs.argmax().item()
            raw_label = model_b.config.id2label[idx]
            conf = probs[0][idx].item()
            
            if "FAKE" in raw_label.upper() or "GENERATED" in raw_label.upper():
                label = "FAKE"
            else:
                label = "REAL"

        return {
            "label": label,
            "confidence": round(conf * 100, 2),
            "router_decision": {
                "sharpness_score": round(sharpness, 2),
                "threshold": SHARPNESS_THRESHOLD,
                "model_used": used_model
            }
        }

    except Exception as e:
        print(e)
        return {"error": str(e)}