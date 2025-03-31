import torch
import pandas as pd
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Charger modèle
model = SentimentClassifier(n_classes=len(CLASS_NAMES))
model.load_state_dict(torch.load("src/best_model.bin", map_location=device))
model = model.to(device)

# Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

# Charger et préparer le test set
df = pd.read_csv("dataset.csv")
df["sentiment"] = df.score.apply(
    lambda x: 0 if x <= 2 else 1 if x == 3 else 2
)

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
    df_test, tokenizer, MAX_LEN, BATCH_SIZE
)

loss_fn = torch.nn.CrossEntropyLoss().to(device)

# Évaluation
acc, loss, preds, targets = eval_model(
    model, test_loader, loss_fn, device
)

# Affichage du score (GitHub Actions lit cette ligne !)
print(f"Test accuracy: {acc:.4f}")
