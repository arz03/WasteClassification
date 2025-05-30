import torch
import timm
import torch.nn as nn
import cv2
from torchvision import transforms
from PIL import Image


# Define the WasteClassifier model (matching the notebook version)
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


def list_cameras(max_cams=10):
    available_cams = []
    for i in range(max_cams):  # Test up to `max_cams` indexes
        cap = cv2.VideoCapture(i)
        if cap.read()[0]:  # Check if the camera opens
            available_cams.append(i)
            cap.release()
    return available_cams


# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state_dict = torch.load("best_model.pth", map_location=device)
# Load the best model for evaluation
model = WasteClassifier(num_classes=4)  # Make sure to use the same architecture
model.load_state_dict(state_dict)
model.to(device)
model.eval()  # Set to evaluation mode

print("Model loaded successfully!")

# Define image transformations (resize to model input size)
transform = transforms.Compose([
    transforms.Resize((260, 260)),  # Resize to match EfficientNet B2 input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define class labels (Change as per your dataset)
# class_labels = ["Plastic", "Metal", "Paper", "Glass"]



# def classify_image(image_path):
#     # Read the image
#     # image = cv2.imread(image_path)
#     # if image is None:
#     #     print(f"Error: Unable to read image from {image_path}")
#     #     return
    
#     image = Image.open(image_path).convert("RGB")
#     transformed_image = transform(image).unsqueeze(0).to(device)

#     # Convert BGR to RGB
#     # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     # Preprocess the image
#     img = transformed_image  # Add batch dimension

#     # Make prediction
#     with torch.no_grad():
#         outputs = model(img)
#         print(class_labels)
#         print(outputs)
#         _, predicted = torch.max(outputs, 1)

#     label = class_labels[predicted.item()]
#     print(f"Prediction: {label}")

# # Example usage
# testing_class = 'metal'
# image_path = f'dataset\\{testing_class}\\{testing_class}_020.jpg'
# print(image_path)
# classify_image(image_path)


















# Define class labels (Make sure this matches your label_map from training)
class_labels = ["glass", "metal", "paper", "plastic"]  # Alphabetical order as in training


def predict_with_confidence(model, image_tensor, device, threshold=0.5):
    """Make prediction with confidence score"""
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        if confidence.item() < threshold:
            return "Uncertain", confidence.item()
        else:
            return class_labels[int(predicted.item())], confidence.item()

def predict_with_all_probabilities(model, image_tensor, device):
    """Get all class probabilities for debugging"""
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        
        print("All class probabilities:")
        for i, (cls, prob) in enumerate(zip(class_labels, probabilities[0])):
            print(f"{i}: {cls} -> {prob:.4f}")
        
        confidence, predicted = torch.max(probabilities, 1)
        return class_labels[int(predicted.item())], confidence.item()
    
# app.py - Main application for waste classification using webcam input
# Step 1: Detect available cameras
cameras = list_cameras()

if not cameras:
    print("No cameras detected!")
else:
    print("\nAvailable Cameras:")
    for idx, cam in enumerate(cameras):
        cap = cv2.VideoCapture(cam)
        print(f"{idx}: Camera {cam}")

    # Step 2: Ask user to select a camera
    cam_idx = int(input("\nEnter the camera index to use: "))

    if cam_idx not in range(len(cameras)):
        print("Invalid selection! Exiting.")
    else:
        selected_cam = cameras[cam_idx]
        print(f"\nUsing Camera {selected_cam}")

        # Step 3: Open the selected camera
        cap = cv2.VideoCapture(selected_cam)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame!")
                break

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert the frame to PIL Image
            frame_pil = Image.fromarray(frame_rgb)

            # Preprocess the frame
            img_tensor = transform(frame_pil).unsqueeze(0).to(device) # type: ignore

            # Make prediction with confidence
            label, confidence = predict_with_confidence(model, img_tensor, device, threshold=0.6)

            # Display the label and confidence on the frame
            text = f"Prediction: {label} ({confidence:.2f})"
            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Show the frame
            cv2.imshow("Waste Classification", frame)

            # Press 'q' to quit & press 'p' to show all probabilities for current frame
            key = cv2.waitKey(1) & 0xFF
            if key == ord('p'):
                print("\n" + "="*50)
                predict_with_all_probabilities(model, img_tensor, device)
                print("="*50)
            elif key == ord('q'):
                break

        # Release resources
        cap.release()
        cv2.destroyAllWindows()
