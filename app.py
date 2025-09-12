# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model
model_path = "class_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()
id2label = model.config.id2label

# Create FastAPI app
app = FastAPI(title="Text Classification API")

# Define request body
class TextRequest(BaseModel):
    text: str

@app.post("/predict")
def predict(request: TextRequest):
    text = request.text
    if not text:
        return {"error": "No text provided."}
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        predicted_class_id = logits.argmax(dim=1).item()
        predicted_label = id2label[predicted_class_id]
    return {"prediction": predicted_label}
