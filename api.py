from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
import io
import numpy as np
from typing import List, Dict
import uvicorn

app = FastAPI(title="Waste Classification API", description="API for classifying waste into 4 categories")

# Define the WasteClassifier model (same as in app.py)
class WasteClassifier(nn.Module):
    def __init__(self, num_classes=4, model_name="efficientnet_b2", freeze_base=False):
        super(WasteClassifier, self).__init__()
        
        # Load EfficientNet model dynamically
        self.base_model = timm.create_model(model_name, pretrained=True, num_classes=0)  
        # Remove head - don't create features wrapper
        
        # Get feature extractor output size
        enet_out_size = self.base_model.num_features
        
        # Unfreeze the last few layers for fine-tuning
        if freeze_base:
            # Freeze most layers but keep last few unfrozen
            for param in list(self.base_model.parameters())[:-20]:
                param.requires_grad = False

        # Improved classifier with more regularization (matching notebook)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.BatchNorm1d(enet_out_size),
            nn.Dropout(0.4),
            nn.Linear(enet_out_size, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.base_model.forward_features(x)  # Use forward_features instead of features
        return self.classifier(x)

# Global variables for model and transforms
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
transform = None
class_labels = ["glass", "metal", "paper", "plastic"]  # Alphabetical order as in training

def load_model():
    """Load the trained model"""
    global model, transform
    
    try:
        # Load model
        state_dict = torch.load("best_model.pth", map_location=device)
        model = WasteClassifier(num_classes=4)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        # Define image transformations (same as in app.py)
        transform = transforms.Compose([
            transforms.Resize((260, 260)),  # Resize to match EfficientNet B2 input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Model loaded successfully on {device}!")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def predict_image(image_tensor: torch.Tensor) -> List[Dict[str, float]]:
    """
    Make prediction and return all class probabilities sorted in descending order
    """
    with torch.no_grad():
        if model is None:
            raise RuntimeError("Model is not loaded. Please check the startup event.")
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        
        # Create list of class predictions with probabilities
        predictions = []
        for i, (cls, prob) in enumerate(zip(class_labels, probabilities[0])):
            predictions.append({
                "class": cls,
                "probability": float(prob.item())
            })
        
        # Sort by probability in descending order
        predictions.sort(key=lambda x: x["probability"], reverse=True)
        
        return predictions

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    success = load_model()
    if not success:
        raise RuntimeError("Failed to load model on startup")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Waste Classification API is running", "device": str(device)}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device),
        "classes": class_labels
    }

@app.post("/predict")
async def predict_waste(file: UploadFile = File(...)):
    """
    Predict waste class from uploaded image
    
    Returns:
        JSON response with all class probabilities sorted in descending order
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image file
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Preprocess image
        image_tensor = transform(image).unsqueeze(0).to(device) # type: ignore
        
        # Make prediction
        predictions = predict_image(image_tensor)
        
        # Get top prediction
        top_prediction = predictions[0]
        
        return JSONResponse(content={
            "success": True,
            "filename": file.filename,
            "predicted_class": top_prediction["class"],
            "confidence": top_prediction["probability"],
            "all_predictions": predictions,
            "device_used": str(device)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Predict waste class for multiple images
    
    Returns:
        JSON response with predictions for all images
    """
    if len(files) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 10 images allowed per batch")
    
    results = []
    
    for file in files:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            results.append({
                "filename": file.filename,
                "success": False,
                "error": "File must be an image"
            })
            continue
        
        try:
            # Read and process image
            image_data = await file.read()
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            
            # Preprocess image
            image_tensor = transform(image).unsqueeze(0).to(device) # type: ignore
            
            # Make prediction
            predictions = predict_image(image_tensor)
            top_prediction = predictions[0]
            
            results.append({
                "filename": file.filename,
                "success": True,
                "predicted_class": top_prediction["class"],
                "confidence": top_prediction["probability"],
                "all_predictions": predictions
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    return JSONResponse(content={
        "success": True,
        "total_images": len(files),
        "results": results,
        "device_used": str(device)
    })

if __name__ == "__main__":
    # open "localhost:8000/docs" in browser to view API documentation and test endpoints
    print("Click the link below to open the API documentation in your browser:")
    print("http://localhost:8000/docs")
    # Run the server
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
