from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

# 1. SETUP API
app = FastAPI(title="DeepFake Detector API")

# Enable CORS (Allows Chrome Extension to talk to this server)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, we'd limit this. For now, allow all.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. LOAD THE MODEL
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    print("Loading model...")
    # Load ResNet18 architecture
    model = models.resnet18(weights=None)
    
    # Replace the final layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    # Load the weights we trained
    try:
        model.load_state_dict(torch.load("resnet18_deepfake.pth", map_location=device))
    except FileNotFoundError:
        raise RuntimeError("Model file 'resnet18_deepfake.pth' not found! Make sure it's in the same folder as main.py")
    
    model = model.to(device)
    model.eval() # Set to evaluation mode
    print(f"Model loaded on {device}!")
    return model

# Initialize model
model = load_model()

# 3. DEFINE PREPROCESSING (UPDATED FOR HIGH-RES MODEL)
# We removed the 32x32 downgrade. Now we keep it sharp.
transform_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 4. API ENDPOINT
@app.get("/")
def home():
    return {"message": "DeepFake Detector API is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a JPEG or PNG.")

    try:
        # Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Preprocess
        image_tensor = transform_pipeline(image).unsqueeze(0) # Add batch dimension
        image_tensor = image_tensor.to(device)
        
        # Inference
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
        # Parse result (Class 0=FAKE, Class 1=REAL)
        fake_prob = probabilities[0][0].item()
        real_prob = probabilities[0][1].item()
        
        if fake_prob > real_prob:
            label = "FAKE"
            confidence = fake_prob
        else:
            label = "REAL"
            confidence = real_prob

        return {
            "label": label,
            "confidence": round(confidence * 100, 2),
            "probabilities": {
                "fake": round(fake_prob * 100, 2),
                "real": round(real_prob * 100, 2)
            }
        }

    except Exception as e:
        return {"error": str(e)}