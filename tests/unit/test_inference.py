import unittest
from unittest.mock import MagicMock, patch

import torch

from src.inference import SentimentAnalyzerApp, SentimentClassifier


class TestSentimentAnalyzerApp(unittest.TestCase):
    def setUp(self):
        self.mock_root = MagicMock()
        self.mock_root.children = {}

        self.tk_patcher = patch(
            "src.inference.tk.Tk", return_value=self.mock_root
        )
        self.messagebox_patcher = patch("src.inference.messagebox")

        self.mock_tokenizer = MagicMock()
        self.mock_model = MagicMock(spec=SentimentClassifier)
        self.mock_model.device = "cpu"
        self.mock_model.to.return_value = self.mock_model

        self.mock_tk = self.tk_patcher.start()
        self.mock_messagebox = self.messagebox_patcher.start()

        self.model_patcher = patch(
            "src.inference.SentimentClassifier",
            return_value=self.mock_model,
        ).start()

        self.tokenizer_patcher = patch(
            "src.inference.BertTokenizer.from_pretrained",
            return_value=self.mock_tokenizer,
        ).start()

        self.torch_load_patcher = patch(
            "src.inference.torch.load", return_value={}
        ).start()

        self.app = SentimentAnalyzerApp(self.mock_root)

        self.app.result_label = MagicMock()
        self.app.text_input = MagicMock()

    def tearDown(self):
        patch.stopall()

    def test_setup_model_success(self):
        """Test successful model initialization"""
        self.assertTrue(hasattr(self.app, "model"))
        self.assertEqual(self.app.model, self.mock_model)
        self.mock_model.eval.assert_called_once()

        self.torch_load_patcher.assert_called_once_with(
            "src/best_model.bin", map_location=self.app.device
        )
        self.mock_root.title.assert_called_once_with("Sentiment Analyzer")

    def test_setup_model_failure(self):
        """Test model loading failure handling"""
        with patch(
            "src.inference.BertTokenizer.from_pretrained",
            side_effect=Exception("Load error"),
        ):
            app = SentimentAnalyzerApp(self.mock_root)

            args, _ = self.mock_messagebox.showerror.call_args
            self.assertEqual(args[0], "Error")
            self.assertIn("Load error", args[1])

            self.mock_root.destroy.assert_called_once()
            self.assertFalse(hasattr(app, "model"))

    def test_empty_input(self):
        """Test empty input validation"""
        self.app.text_input.get.return_value = ""
        self.app.analyze_sentiment()

        self.mock_messagebox.showwarning.assert_called_once_with(
            "Warning", "Please enter some text!"
        )
        self.app.result_label.config.assert_not_called()

    @patch("src.inference.F.softmax")
    def test_sentiment_analysis_positive(self, mock_softmax):
        """Test successful sentiment analysis"""
        test_text = "I love this product!"
        mock_softmax.return_value = torch.tensor([[0.1, 0.2, 0.7]])

        self.app.text_input.get.return_value = test_text
        self.mock_tokenizer.encode_plus.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

        self.app.analyze_sentiment()

        self.mock_tokenizer.encode_plus.assert_called_once_with(
            test_text,
            max_length=160,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        self.app.result_label.config.assert_called_once_with(
            text="Sentiment: Positive (Confidence: 0.70)",
            fg="#2ecc71"
        )

    def test_color_mapping(self):
        """Test sentiment-color mapping"""
        self.assertEqual(self.app.get_color("Positive"), "#2ecc71")
        self.assertEqual(self.app.get_color("Neutral"), "#f1c40f")
        self.assertEqual(self.app.get_color("Negative"), "#e74c3c")
        self.assertEqual(self.app.get_color("Unknown"), "black")


if __name__ == "__main__":
    unittest.main()