from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import uvicorn

app = FastAPI()

# Load model
model_path = "class_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()
id2label = model.config.id2label

class InputText(BaseModel):
    text: str

@app.post("/predict")
def predict(input: InputText):
    inputs = tokenizer(input.text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        predicted_class_id = logits.argmax(dim=1).item()
        return {"prediction": id2label[predicted_class_id]}