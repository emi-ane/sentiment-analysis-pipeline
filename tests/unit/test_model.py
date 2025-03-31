import os
import sys

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from src.model import (
    BATCH_SIZE,
    CLASS_NAMES,
    MAX_LEN,
    MODEL_NAME,
    SentimentClassifier,
    SentimentDataset,
    create_data_loader,
)

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.insert(0, project_root)


@pytest.fixture
def tokenizer():
    return BertTokenizer.from_pretrained(MODEL_NAME)


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "content": [
                "This is a positive review!",
                "Negative experience :(",
                "Neutral comment.",
                "Another positive one!",
                "Not great, not terrible.",
                "Excellent service!",
                "Would not recommend",
                "Average performance",
                "Best product ever",
                "Worst purchase of my life",
                "Okay experience",
                "Highly recommended",
                "Disappointing quality",
                "Above expectations",
                "Below average",
                "Mixed feelings",
            ],
            "sentiment": [
                2, 0, 1, 2, 1, 2, 0, 1,
                2, 0, 1, 2, 0, 2, 1, 1
            ],
        }
    )


@pytest.fixture
def sample_dataset(sample_data, tokenizer):
    return SentimentDataset(
        texts=sample_data.content.to_numpy(),
        targets=sample_data.sentiment.to_numpy(),
        tokenizer=tokenizer,
        max_len=MAX_LEN,
    )


@pytest.fixture
def model():
    return SentimentClassifier(len(CLASS_NAMES))


def test_dataset_item(sample_dataset):
    item = sample_dataset[0]
    assert isinstance(item, dict)
    assert "input_ids" in item
    assert "attention_mask" in item
    assert "targets" in item
    assert item["input_ids"].shape == (MAX_LEN,)
    assert item["attention_mask"].shape == (MAX_LEN,)
    assert isinstance(item["targets"], torch.Tensor)


def test_dataset_length(sample_dataset, sample_data):
    assert len(sample_dataset) == len(sample_data)


def test_model_initialization(model):
    assert isinstance(model, SentimentClassifier)
    assert hasattr(model, "bert")
    assert hasattr(model, "drop")
    assert hasattr(model, "out")
    assert isinstance(model.drop, torch.nn.Dropout)
    assert model.drop.p == 0.3
    assert isinstance(model.out, torch.nn.Linear)
    assert model.out.out_features == len(CLASS_NAMES)


def test_model_forward_pass(model, sample_dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    sample = sample_dataset[0]
    input_ids = sample["input_ids"].unsqueeze(0).to(device)
    attention_mask = sample["attention_mask"].unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_ids, attention_mask)

    assert output.shape == (1, len(CLASS_NAMES))
    assert torch.is_tensor(output)


def test_data_loader_creation(sample_data, tokenizer):
    data_loader = create_data_loader(
        sample_data, tokenizer, MAX_LEN, BATCH_SIZE
    )

    assert isinstance(data_loader, DataLoader)
    assert data_loader.batch_size == BATCH_SIZE

    batch = next(iter(data_loader))
    expected_batch_size = min(BATCH_SIZE, len(sample_data))

    assert batch["input_ids"].shape == (expected_batch_size, MAX_LEN)
    assert batch["attention_mask"].shape == (expected_batch_size, MAX_LEN)
    assert batch["targets"].shape == (expected_batch_size,)


def test_training_step(model, sample_dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    batch = {
        "input_ids": torch.stack([
            sample_dataset[0]["input_ids"],
            sample_dataset[1]["input_ids"]
        ]).to(device),
        "attention_mask": torch.stack([
            sample_dataset[0]["attention_mask"],
            sample_dataset[1]["attention_mask"]
        ]).to(device),
        "targets": torch.tensor([
            sample_dataset[0]["targets"],
            sample_dataset[1]["targets"]
        ]).to(device),
    }

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    outputs = model(batch["input_ids"], batch["attention_mask"])
    loss = loss_fn(outputs, batch["targets"])
    initial_loss = loss.item()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    outputs_new = model(batch["input_ids"], batch["attention_mask"])
    loss_new = loss_fn(outputs_new, batch["targets"])

    assert loss_new.item() != initial_loss


def test_model_save_load(tmp_path, model):
    save_path = tmp_path / "test_model.bin"
    torch.save(model.state_dict(), save_path)

    loaded_model = SentimentClassifier(len(CLASS_NAMES))
    loaded_model.load_state_dict(torch.load(save_path))

    for (name, param), (loaded_name, loaded_param) in zip(
        model.named_parameters(),
        loaded_model.named_parameters(),
    ):
        assert name == loaded_name
        assert torch.equal(param, loaded_param)


def test_data_loader_shuffling(sample_data, tokenizer):
    data_loader = create_data_loader(
        sample_data, tokenizer, MAX_LEN, BATCH_SIZE
    )
    first_batch = next(iter(data_loader))["input_ids"]
    second_batch = next(iter(data_loader))["input_ids"]

    assert not torch.equal(first_batch, second_batch)
