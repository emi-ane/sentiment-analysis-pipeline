from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from transformers import BertTokenizer
from src.model import SentimentClassifier, CLASS_NAMES

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
model = SentimentClassifier(n_classes=len(CLASS_NAMES))
model.load_state_dict(torch.load("src/best_model.bin", map_location=device))
model.to(device)
model.eval()


class TextInput(BaseModel):
    text: str


@app.post("/predict")
def predict(data: TextInput):
    encoding = tokenizer.encode_plus(
        data.text,
        max_length=160,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probs = F.softmax(outputs, dim=1)
        confidence, prediction = torch.max(probs, dim=1)

    sentiment = CLASS_NAMES[prediction.item()]
    return {
        "sentiment": sentiment,
        "confidence": round(confidence.item(), 4)
    }

# 👉 Nouvelle version sans appel à index.html
@app.get("/")
def root():
    return {"message": "Bienvenue sur l'API de prédiction de sentiment ✨"}
