# 📌 Sentiment Analysis Pipeline

## 📝 Description

This project implements a sentiment analysis pipeline using a BERT-based NLP model. It is designed to classify text as either **positive** or **negative**, leveraging state-of-the-art deep learning techniques. The solution is modular, testable, and reproducible.

---

## ⚙️ Installation

### 1. Clone the Repository

*** bash
git clone https://github.com/ton-repo/sentiment-analysis-pipeline.git
cd sentiment-analysis-pipeline 

### 2. Create a Virtual Environment

*** bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

### 3. Install Dependencies
*** bash
pip install -r requirements.txt

## 📊 Usage
All main scripts are located in the src/ folder.

### 1. Data Extraction
Loads raw CSV data and handles common errors such as:
  - FileNotFoundError;
  - PermissionError;
  - UnicodeDecodeError

*** bash
python src/data_extraction.py

### 2. Data Processing
Cleans and preprocesses the loaded text: 
  - Removes special characters;
  - Lowercases the text;
  - Tokenizes using bert-base-uncased;
  - Splits the dataset into training and validation sets

*** bash
python src/data_processing.py

### 3. Model Training
Fine-tunes a pretrained BERT model (AutoModelForSequenceClassification) on the sentiment dataset using Hugging Face's Trainer API or a custom loop.

*** bash
python src/model.py

### 4. Inference
Makes sentiment predictions on new input text.

*** bash
python src/inference.py

## ✅ Testing
Unit tests are available in the tests/unit/ folder to verify each module independently:
  - test_data_extraction.py — Tests data loading and error handling
  - test_data_processing.py — Tests text cleaning and tokenization
  - test_model.py — Tests model instantiation and dummy training pass
  - test_inference.py — Tests the end-to-end inference process

Run tests using pytest

## 🔧 Challenges Faced and Collaboration
### ⚠️ Git sizes
Some of the files involving the model creation were to heavy and were difficult to put on the plateform 

### 🐢 Performance constraints
The number of training epochs from the original Kaggle notebook had to be reduced to fit our hardware capabilities.

### 👥 Collaboration
Git workflow with feature branches, pull requests, and code reviews

Each team member handled specific parts of the pipeline (Cassandra and Noemi handled the data extraction/Processing and the corresponding Testing files and Dhaval handled the model and inference parts)

Commit messages were kept descriptive and task-specific

### 🔗 Resources
Kaggle Inspiration Notebook: Sentiment Analysis Using BERT

## 🧑‍💻 Contributors
  - Noémi DOMBOU (emi-ane)
  - Dhavalkumar PATEL (Dhavalpatel1811)
  - Cassandra NONGO (Cassydy-prog)
