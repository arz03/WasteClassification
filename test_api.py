import requests
import json
import os

# API endpoint
API_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint"""
    print("=== Testing Health Check ===")
    response = requests.get(f"{API_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_prediction(image_path):
    """Test prediction on a single image"""
    print(f"=== Testing Prediction for {image_path} ===")
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    with open(image_path, 'rb') as f:
        files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
        response = requests.post(f"{API_URL}/predict", files=files)
    
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Predicted Class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print("All Predictions (sorted by probability):")
        for pred in result['all_predictions']:
            print(f"  {pred['class']}: {pred['probability']:.4f}")
    else:
        print(f"Error: {response.text}")
    print()

def test_batch_prediction(image_paths):
    """Test batch prediction on multiple images"""
    print("=== Testing Batch Prediction ===")
    
    files = []
    for image_path in image_paths:
        if os.path.exists(image_path):
            files.append(('files', (os.path.basename(image_path), open(image_path, 'rb'), 'image/jpeg')))
        else:
            print(f"Image not found: {image_path}")
    
    if not files:
        print("No valid images found for batch testing")
        return
    
    response = requests.post(f"{API_URL}/predict/batch", files=files)
    
    # Close file handles
    for _, (_, file_handle, _) in files:
        file_handle.close()
    
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Total Images: {result['total_images']}")
        print("Results:")
        for res in result['results']:
            if res['success']:
                print(f"  {res['filename']}: {res['predicted_class']} ({res['confidence']:.4f})")
            else:
                print(f"  {res['filename']}: ERROR - {res['error']}")
    else:
        print(f"Error: {response.text}")
    print()

if __name__ == "__main__":
    # Test health check
    test_health_check()
    
    # Test single prediction - using sample images from dataset
    test_images = [
        "dataset/glass/glass_001.jpg",
        "dataset/metal/metal_001.jpg", 
        "dataset/paper/paper_001.jpg",
        "dataset/plastic/plastic_001.jpg"
    ]
    
    # Test individual predictions
    for image_path in test_images:
        test_prediction(image_path)
    
    # Test batch prediction
    test_batch_prediction(test_images)
