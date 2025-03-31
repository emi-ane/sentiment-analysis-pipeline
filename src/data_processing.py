import re

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer


# Function to convert score to sentiment
def to_sentiment(rating):

    rating = int(rating)

    # Convert to class
    if rating <= 2:
        return 0
    elif rating == 3:
        return 1
    else:
        return 2


# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Set the model name
MODEL_NAME = "bert-base-cased"

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)


# Text cleaning and preprocessing
def clean_text(text):
    """
    Cleans and preprocesses the input text.
    - Removes unnecessary characters.
    - Converts text to lowercase.
    - Normalizes whitespace.
    """
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters and numbers
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"\s+", " ", text).strip()  # Normalize whitespace
    return text


# Function to add a sentiment column if missing
def add_sentiment_column(df):
    """
    Adds a 'sentiment' column based on the 'score' column.
    - score >= 4 → positive (1)
    - score < 4 → negative (0)
    """
    if "score" in df.columns:
        df["sentiment"] = df["score"].apply(lambda x: 1 if x >= 4 else 0)
    else:
        raise ValueError(
            "Error: The dataset must contain either 'score' or 'sentiment' columns."
        )

    return df


# Custom Dataset class for reviews
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


# Function to create a DataLoader
def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = GPReviewDataset(
        reviews=df["content"].to_numpy(),
        targets=df["sentiment"].to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len,
    )
    return DataLoader(ds, batch_size=batch_size, num_workers=0)


# Main function for data processing
def process_data(df):
    """
    Processes the input DataFrame:
    - Cleans the text.
    - Ensures 'sentiment' column exists.
    - Splits the data into training, validation, and test sets.
    - Tokenizes the text.
    - Creates DataLoaders.
    """
    # Check if 'content' and 'sentiment' exist
    if "content" not in df.columns:
        raise ValueError("Error: The DataFrame must contain a 'content' column.")

    # If 'sentiment' is missing, add it from 'score'
    if "sentiment" not in df.columns:
        df = add_sentiment_column(df)

    # Clean the text
    df["content"] = df["content"].astype(str).apply(clean_text)

    # Split the data into training, validation, and test sets
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)
    df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=RANDOM_SEED)

    print(f"Training set size: {df_train.shape}")
    print(f"Validation set size: {df_val.shape}")
    print(f"Test set size: {df_test.shape}")

    # Tokenize and create DataLoaders
    max_len = 128
    batch_size = 16

    train_data_loader = create_data_loader(df_train, tokenizer, max_len, batch_size)
    val_data_loader = create_data_loader(df_val, tokenizer, max_len, batch_size)
    test_data_loader = create_data_loader(df_test, tokenizer, max_len, batch_size)

    return train_data_loader, val_data_loader, test_data_loader


# Example usage
if __name__ == "__main__":
    dataset_path = r"C:\Users\noemi\sentiment-analysis-pipeline\dataset.csv"
    df = pd.read_csv(dataset_path)

    # Apply to the dataset
    df["sentiment"] = df.score.apply(to_sentiment)

    # Display initial columns
    print("Initial Columns:", df.columns)

    # Add sentiment column if missing
    if "sentiment" not in df.columns:
        df = add_sentiment_column(df)

    # Display final columns after correction
    print("Final Columns:", df.columns)

    # Process the data
    train_loader, val_loader, test_loader = process_data(df)

    # Inspect the first batch of the training DataLoader
    data = next(iter(train_loader))
    print(data.keys())
    print(data["input_ids"].shape)
    print(data["attention_mask"].shape)
    print(data["targets"].shape)
