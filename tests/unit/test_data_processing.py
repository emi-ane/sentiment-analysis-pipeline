import unittest

import pandas as pd
import torch
from transformers import BertTokenizer

from src.data_processing import (
    clean_text,
    create_data_loader,
)

MODEL_NAME = "bert-base-cased"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)


class TestDataProcessing(unittest.TestCase):

    def test_clean_text(self):
        """Test text cleaning: removes symbols and lowers case."""

        raw_text = "Hello, WORLD!! 123"
        cleaned_text = clean_text(raw_text)
        expected_text = "hello world"
        self.assertEqual(cleaned_text, expected_text)

    def test_tokenization(self):
        """Test if tokenization returns expected token IDs."""
        sample_text = "Hello world"
        encoding = tokenizer.encode_plus(
            sample_text,
            add_special_tokens=True,
            max_length=10,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        expected_tokens = tokenizer.encode(
            sample_text, add_special_tokens=True
        )

        self.assertTrue(
            torch.equal(
                encoding["input_ids"][0][:len(expected_tokens)],
                torch.tensor(expected_tokens),
            )
        )

    def test_create_data_loader(self):
        """Test if DataLoader correctly wraps the dataset."""
        df = pd.DataFrame(
            {
                "content": [
                    "This is a positive review",
                    "This is a negative review",
                ],
                "sentiment": [1, 0],
            }
        )

        max_len = 128
        batch_size = 2
        data_loader = create_data_loader(
            df, tokenizer, max_len, batch_size
        )

        batch = next(iter(data_loader))

        self.assertEqual(batch["input_ids"].shape, (batch_size, max_len))
        self.assertEqual(batch["attention_mask"].shape, (batch_size, max_len))
        self.assertEqual(batch["targets"].shape, (batch_size,))


if __name__ == "__main__":
    unittest.main()
