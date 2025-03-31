import re

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer


RANDOM_SEED = 42
MODEL_NAME = "bert-base-cased"

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# -------------------- Utilitaires --------------------


def to_sentiment(rating):
    """
    Convertit une note numérique en classe de sentiment :
    - 0 = négatif
    - 1 = neutre
    - 2 = positif
    """
    rating = int(rating)

    if rating <= 2:
        return 0
    elif rating == 3:
        return 1
    return 2


def clean_text(text):
    """
    Nettoie et prétraite un texte :
    - Retire les caractères spéciaux
    - Met en minuscules
    - Normalise les espaces
    """
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def add_sentiment_column(df):
    """
    Ajoute une colonne 'sentiment' si elle n'existe pas.
    Classe les scores :
    - >= 4 → 1 (positif)
    - < 4 → 0 (négatif)
    """
    if "score" in df.columns:
        df["sentiment"] = df["score"].apply(lambda x: 1 if x >= 4 else 0)
    else:
        raise ValueError(
            "Dataset must contain either 'score' or 'sentiment' columns."
        )
    return df



class GPReviewDataset(Dataset):
    def __init__(self, reviews, targets, tokenizer, max_len):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review = str(self.reviews[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "review_text": review,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "targets": torch.tensor(target, dtype=torch.long),
        }



def create_data_loader(df, tokenizer, max_len, batch_size):
    dataset = GPReviewDataset(
        reviews=df["content"].to_numpy(),
        targets=df["sentiment"].to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len,
    )
    return DataLoader(dataset, batch_size=batch_size, num_workers=0)



def process_data(df):
    """
    Prétraite les données :
    - Vérifie les colonnes
    - Nettoie les textes
    - Divise en sets train/val/test
    - Retourne les DataLoaders
    """
    if "content" not in df.columns:
        raise ValueError(
            "'content' column missing in DataFrame."
        )

    if "sentiment" not in df.columns:
        df = add_sentiment_column(df)

    df["content"] = df["content"].astype(str).apply(clean_text)

    df_train, df_temp = train_test_split(
        df, test_size=0.2, random_state=RANDOM_SEED
    )
    df_val, df_test = train_test_split(
        df_temp, test_size=0.5, random_state=RANDOM_SEED
    )

    print(f"Training set size: {df_train.shape}")
    print(f"Validation set size: {df_val.shape}")
    print(f"Test set size: {df_test.shape}")

    max_len = 128
    batch_size = 16

    train_loader = create_data_loader(df_train, tokenizer, max_len, batch_size)
    val_loader = create_data_loader(df_val, tokenizer, max_len, batch_size)
    test_loader = create_data_loader(df_test, tokenizer, max_len, batch_size)

    return train_loader, val_loader, test_loader
