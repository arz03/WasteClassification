# WasteClassification

Image classification model for waste segregation using PyTorch. This fine-tuned EfficientNet 2b model classifies images into four categories: metal, paper, glass, and plastic.

- **Framework:** PyTorch  
- **Model:** EfficientNet 2b (fine-tuned)  
- **Classes:** metal, paper, glass, plastic  
- **Accuracy:** 94% on training data  
- **Dataset:** Imbalanced dataset handled by manually adjusting class weights  
- **API:** FastAPI integration for easy model serving

## Quick Start

### 1. Install Dependencies

PyTorch must be installed manually with CUDA support.

Run this before installing the rest:
```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
```
Then install dependencies using:
```bash
pip install -r requirements.txt
```

### 2. Run the API Server

```bash
python api.py
```

The server will start on `http://127.0.0.1:5001/`.

### 3. Example API Usage

Send an image for classification using `requests`:

```python
import requests

url = "http://127.0.0.1:5001/predict"
files = {'file': open('example.jpg', 'rb')}
response = requests.post(url, files=files)
print(response.json())  # {'class': 'plastic'}
```

Alternatively, you can use `test_api.py` to test the API:

```bash
python test_api.py
```
