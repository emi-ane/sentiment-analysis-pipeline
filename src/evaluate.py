import torch
import pandas as pd
import numpy as np
import json
import sys
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

from model import (
    SentimentClassifier,
    create_data_loader,
    eval_model,
    CLASS_NAMES,
    MAX_LEN,
    BATCH_SIZE,
    RANDOM_SEED,
)

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "src/best_model.bin"
DATASET_PATH = "dataset.csv"
THRESHOLD = 0.70

# Charger le modèle
model = SentimentClassifier(n_classes=len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)

# Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

# Charger et préparer le jeu de test
df = pd.read_csv(DATASET_PATH)
df["sentiment"] = df.score.apply(lambda x: 0 if x <= 2 else 1 if x == 3 else 2)

_, df_test = train_test_split(
    df,
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=df.sentiment,
)
_, df_test = train_test_split(
    df_test,
    test_size=0.5,
    random_state=RANDOM_SEED,
    stratify=df_test.sentiment,
)

test_loader = create_data_loader(
    df_test,
    tokenizer,
    MAX_LEN,
    BATCH_SIZE
)

loss_fn = torch.nn.CrossEntropyLoss().to(device)

# Évaluation
acc, loss, preds, targets = eval_model(model, test_loader, loss_fn, device)

# Affichage
print(f"Test accuracy: {acc:.4f}")
print(f"Test loss: {loss:.4f}")

# Sauvegarde des métriques
metrics = {
    "accuracy": acc.item(),
    "loss": loss
}
with open("metrics.json", "w") as f:
    json.dump(metrics, f)

# Échec du job si performance insuffisante
if acc < THRESHOLD:
    print(f"❌ Accuracy trop basse ({acc:.4f}) — échec du workflow.")
    sys.exit(1)
else:
    print(f"✅ Accuracy suffisante ({acc:.4f}) — OK.")
