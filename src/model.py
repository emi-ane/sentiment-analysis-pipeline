# interface.py
import torch
import tkinter as tk
from tkinter import messagebox
from transformers import BertTokenizer
from torch import nn
import torch.nn.functional as F  # <-- Added this import
from transformers import BertModel

class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.last_hidden_state[:, 0, :]
        return self.out(self.drop(pooled_output))

class SentimentAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = ["Negative", "Neutral", "Positive"]
        self.setup_model()
    
    def setup_model(self):
        try:
            self.tokenizer = BertTokenizer.from_pretrained('tokenizer')
            self.model = SentimentClassifier(len(self.class_names)).to(self.device)
            self.model.load_state_dict(torch.load('best_model.bin', map_location=self.device))
            self.model.eval()
        except Exception as e:
            messagebox.showerror("Error", f"Model loading failed: {str(e)}")
            self.root.destroy()
            return  # Return here to prevent calling create_widgets if model loading fails
        
        self.create_widgets()
    
    def create_widgets(self):
        self.root.title("Sentiment Analyzer")
        
        self.text_input = tk.Text(self.root, height=8, width=50, font=("Arial", 12))
        self.text_input.pack(pady=10, padx=10)
        
        analyze_btn = tk.Button(
            self.root, 
            text="Analyze Sentiment", 
            command=self.analyze_sentiment,
            font=("Arial", 12),
            bg="#4CAF50",
            fg="white"
        )
        analyze_btn.pack(pady=5)
        
        self.result_label = tk.Label(
            self.root, 
            text="", 
            font=("Arial", 14, "bold"),
            pady=10
        )
        self.result_label.pack()
    
    def analyze_sentiment(self):
        text = self.text_input.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Warning", "Please enter some text!")
            return
        
        try:
            encoding = self.tokenizer.encode_plus(
                text,
                max_length=160,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask)
                probabilities = F.softmax(outputs, dim=1)
                confidence, prediction = torch.max(probabilities, dim=1)
            
            sentiment = self.class_names[prediction.item()]  # Added .item() to convert tensor to Python scalar
            self.result_label.config(
                text=f"Sentiment: {sentiment} (Confidence: {confidence.item():.2f})",
                fg=self.get_color(sentiment)
            )
        
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
    
    def get_color(self, sentiment):
        return {
            "Negative": "#e74c3c",
            "Neutral": "#f1c40f",
            "Positive": "#2ecc71"
        }.get(sentiment, "black")

if __name__ == '__main__':
    root = tk.Tk()
    app = SentimentAnalyzerApp(root)
    root.mainloop()