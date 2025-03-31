import logging
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import (AdamW, BertModel, BertTokenizer,
                          get_linear_schedule_with_warmup)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()],
)

# Configuration
RANDOM_SEED = 42
MAX_LEN = 160
BATCH_SIZE = 16
EPOCHS = 5  # Increased for better training
LEARNING_RATE = 2e-5
MODEL_NAME = "bert-base-cased"
CLASS_NAMES = ["Negative", "Neutral", "Positive"]
SAVE_DIR = "model_artifacts"

# Create save directory if it doesn't exist
os.makedirs(SAVE_DIR, exist_ok=True)

# Set random seeds for reproducibility
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")


class SentimentDataset(Dataset):
    def __init__(self, texts, targets, tokenizer, max_len):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "targets": torch.tensor(target, dtype=torch.long),
        }


class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.bert = BertModel.from_pretrained(MODEL_NAME)
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        output = self.drop(pooled_output)
        return self.out(output)


def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = SentimentDataset(
        texts=df.content.to_numpy(),
        targets=df.sentiment.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


def train_epoch(model, data_loader, loss_fn, optimizer, scheduler, device):
    model.train()
    losses = []
    correct_predictions = 0
    all_predictions = []
    all_targets = []

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs, targets)

        _, preds = torch.max(outputs, dim=1)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        correct_predictions += torch.sum(preds == targets)
        all_predictions.extend(preds.cpu().tolist())
        all_targets.extend(targets.cpu().tolist())

    return (
        correct_predictions.double() / len(data_loader.dataset),
        np.mean(losses),
        all_predictions,
        all_targets,
    )


def eval_model(model, data_loader, loss_fn, device):
    model.eval()
    losses = []
    correct_predictions = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["targets"].to(device)

            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, targets)

            _, preds = torch.max(outputs, dim=1)

            losses.append(loss.item())
            correct_predictions += torch.sum(preds == targets)
            all_predictions.extend(preds.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())

    return (
        correct_predictions.double() / len(data_loader.dataset),
        np.mean(losses),
        all_predictions,
        all_targets,
    )


def train():
    # Load and prepare data
    try:
        df = pd.read_csv("../dataset.csv")
        logging.info(f"Loaded dataset with {len(df)} samples")
    except FileNotFoundError:
        logging.error("Dataset file 'dataset.csv' not found!")
        return

    # Check if dataset has the expected columns
    required_columns = ["content", "score"]
    if not all(col in df.columns for col in required_columns):
        logging.error(
            f"Dataset missing required columns. "
            f"Expected: {required_columns}, Got: {list(df.columns)}"
        )
        return

    # Convert scores to sentiment classes
    df["sentiment"] = df.score.apply(lambda x: 0 if x <= 2 else 1 if x == 3 else 2)

    # Display class distribution
    class_counts = df.sentiment.value_counts()
    logging.info(f"Class distribution: {class_counts.to_dict()}")

    # Split data
    df_train, df_test = train_test_split(
        df, test_size=0.2, random_state=RANDOM_SEED, stratify=df.sentiment
    )
    df_val, df_test = train_test_split(
        df_test, test_size=0.5, random_state=RANDOM_SEED, stratify=df_test.sentiment
    )

    logging.info(f"Train set: {len(df_train)} samples")
    logging.info(f"Validation set: {len(df_val)} samples")
    logging.info(f"Test set: {len(df_test)} samples")

    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = SentimentClassifier(len(CLASS_NAMES)).to(device)

    # Create data loaders
    train_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
    val_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
    test_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

    # Training setup
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, correct_bias=False)
    loss_fn = nn.CrossEntropyLoss().to(device)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # Training loop
    best_accuracy = 0
    for epoch in range(EPOCHS):
        logging.info(f"Epoch {epoch + 1}/{EPOCHS}")
        logging.info("-" * 30)

        # Training phase
        train_acc, train_loss, train_preds, train_targets = train_epoch(
            model, train_loader, loss_fn, optimizer, scheduler, device
        )

        # Validation phase
        val_acc, val_loss, val_preds, val_targets = eval_model(
            model, val_loader, loss_fn, device
        )

        logging.info(f"Train loss: {train_loss:.4f} Acc: {train_acc:.4f}")
        logging.info(f"Val loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            # Save model with path
            model_path = os.path.join(SAVE_DIR, "best_model.bin")
            torch.save(model.state_dict(), model_path)
            logging.info(f"Saved new best model to {model_path}")

    # Final evaluation on test set
    test_acc, test_loss, test_preds, test_targets = eval_model(
        model, test_loader, loss_fn, device
    )
    logging.info(f"Test loss: {test_loss:.4f} Acc: {test_acc:.4f}")

    # Print detailed classification report
    report = classification_report(
        test_targets, test_preds, target_names=CLASS_NAMES, digits=4
    )
    logging.info("Classification Report:\n" + report)

    # Save tokenizer
    tokenizer_path = os.path.join(SAVE_DIR, "tokenizer")
    tokenizer.save_pretrained(tokenizer_path)

    # Create a model config file
    with open(os.path.join(SAVE_DIR, "model_config.txt"), "w") as f:
        f.write(f"Model name: {MODEL_NAME}\n")
        f.write(f"Max sequence length: {MAX_LEN}\n")
        f.write(f"Classes: {', '.join(CLASS_NAMES)}\n")
        f.write(f"Train accuracy: {train_acc:.4f}\n")
        f.write(f"Validation accuracy: {val_acc:.4f}\n")
        f.write(f"Test accuracy: {test_acc:.4f}\n")

    logging.info(f"Training complete. Model artifacts saved to {SAVE_DIR}")

    # Copy model files to main directory for compatibility with interface.py
    import shutil

    shutil.copy(os.path.join(SAVE_DIR, "best_model.bin"), "best_model.bin")
    if os.path.exists("tokenizer"):
        shutil.rmtree("tokenizer")
    shutil.copytree(tokenizer_path, "tokenizer")
    logging.info("Copied model files to main directory for interface compatibility")


if __name__ == "__main__":
    train()
