import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

label_names = ["sea","mountain","nature","history",
               "play","shopping","food","family","rain"]

def load_model():
    tokenizer = AutoTokenizer.from_pretrained("mie_bert_model")
    model = AutoModelForSequenceClassification.from_pretrained("mie_bert_model")
    model.eval()
    return tokenizer, model

def predict_labels(text, tokenizer, model, threshold=0.5):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.sigmoid(outputs.logits)[0].numpy()

    scores = {label: float(probs[i]) for i, label in enumerate(label_names)}
    active = [l for l, p in scores.items() if p >= threshold]
    return scores, active
