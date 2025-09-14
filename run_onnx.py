import os, json
import numpy as np
import onnxruntime as ort

from tokenizers import Tokenizer  # very small library, not transformers!

# Load tokenizer
tokenizer = Tokenizer.from_file(r"C:\Users\tomas\OneDrive\Desktop\Git\modelo_gastos\onnx_model\tokenizer.json")

# Load ONNX model
ort_session = ort.InferenceSession(r"C:\Users\tomas\OneDrive\Desktop\Git\modelo_gastos\onnx_model\model.onnx")

# Load labels
with open(r"C:\Users\tomas\OneDrive\Desktop\Git\modelo_gastos\onnx_model\labels.json") as f:
    id2label = {int(k): v for k, v in json.load(f).items()}

text = "videojuego"

# Encode input
encoding = tokenizer.encode(text)
input_ids = np.array([encoding.ids], dtype=np.int64)
attention_mask = np.array([[1] * len(encoding.ids)], dtype=np.int64)

# Run inference
ort_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
logits = ort_session.run(None, ort_inputs)[0]

# Prediction
pred_id = int(np.argmax(logits, axis=1)[0])
pred_label = id2label[pred_id]

print(f"prediction: {pred_label}")
