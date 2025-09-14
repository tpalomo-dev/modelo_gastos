from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch, os, json

model_path = r"C:\Users\tomas\OneDrive\Desktop\Git\modelo_gastos\class_model"
save_path = r"C:\Users\tomas\OneDrive\Desktop\Git\modelo_gastos\onnx_model"
os.makedirs(save_path, exist_ok=True)

# Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Save vocab
tokenizer.save_pretrained(save_path)

# Dummy input
dummy = tokenizer("dummy", return_tensors="pt")

# Export to ONNX
torch.onnx.export(
    model,
    (dummy["input_ids"], dummy["attention_mask"]),
    f"{save_path}/model.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                  "attention_mask": {0: "batch", 1: "seq"}},
    opset_version=14,
)

# Save labels
with open(r"C:\Users\tomas\OneDrive\Desktop\Git\modelo_gastos\labels.json", "w") as f:
    json.dump(model.config.id2label, f)