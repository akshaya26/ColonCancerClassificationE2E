import torch
print(4)
from app.model.model_loader import load_model
print(5)
from app.model.preprocess import preprocess_image
print(6)

_model = None

def get_model():
    global _model
    if _model is None:
        print("Loading model...")
        _model = load_model()
        print("Loading model done...")
    return _model

def predict(image_bytes):
    model = get_model()
    img = preprocess_image(image_bytes)

    with torch.no_grad():
        output = model(img)
        probs = torch.sigmoid(output)
        print(output)
        print(probs)

    confidence = float(probs.item())
    pred = 1 if confidence > 0.5 else 0
    return{
        "label": "Benign" if pred ==1 else "Malignant",
        "confidence" : round(float(probs.max()),4)*100
    }