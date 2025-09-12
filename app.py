# app.py
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model
model_path = "class_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()
id2label = model.config.id2label

st.title("Text Classification")

text = st.text_area("Enter text to classify:")

if st.button("Predict"):
    if text:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            logits = model(**inputs).logits
            predicted_class_id = logits.argmax(dim=1).item()
            st.write("Prediction:", id2label[predicted_class_id])
    else:
        st.warning("Please enter some text!")