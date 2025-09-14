# app.py
import os
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import onnxruntime as ort
import json
from tokenizers import Tokenizer  # very small library, not transformers!
import numpy as np

# Build paths relative to this script
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "onnx_model/model.onnx")
TOKENIZER_DIR = os.path.join(BASE_DIR, "onnx_model/tokenizer.json")
ID2LABEL_DIR = os.path.join(BASE_DIR, "onnx_model/labels.json")

tokenizer = Tokenizer.from_file(TOKENIZER_DIR)
# Load ONNX model
ort_session = ort.InferenceSession(MODEL_DIR)

with open(ID2LABEL_DIR) as f:
    id2label = {int(k): v for k, v in json.load(f).items()}

# Load API key from environment variable (set this on Render dashboard)
API_KEY = os.getenv("Render_API", "changeme")

# Create FastAPI app
app = FastAPI(title="Text Classification API")

# Define request body
class TextRequest(BaseModel):
    text: str

@app.post("/predict")
def predict(request: TextRequest, x_api_key: str = Header(...)):
    # Check API key
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    text = request.text
    if not text:
        return {"error": "No text provided."}
    
    # Encode input
    encoding = tokenizer.encode(text)
    input_ids = [encoding.ids]  # shape: [1, seq_len]
    attention_mask = [[1] * len(encoding.ids)]  # same shape

    # Run inference
    ort_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    logits = ort_session.run(None, ort_inputs)[0]  # returns list of lists

    # Prediction (argmax without numpy)
    pred_id = max(range(len(logits[0])), key=lambda i: logits[0][i])
    
    pred_label = id2label[pred_id]
    
    return {"prediction": pred_label}