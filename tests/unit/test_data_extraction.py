import unittest
import pandas as pd 
import os
from src.data_extraction import load_csv_file

class TestDataExtraction(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Define the dataset path before running tests"""
        cls.dataset_path = r"C:\Users\noemi\sentiment-analysis-pipeline\dataset.csv"

        # Check if the dataset file exists before running tests
        if not os.path.exists(cls.dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {cls.dataset_path}")

    def test_load_real_dataset(self):
        """Test if the dataset loads correctly as a DataFrame."""
        df = load_csv_file(self.dataset_path)

        # Check if the function returned a DataFrame
        self.assertIsInstance(df, pd.DataFrame)

        # Ensure it has at least one row and one column
        self.assertGreater(df.shape[0], 0, "Dataset is empty (no rows).")
        self.assertGreater(df.shape[1], 0, "Dataset has no columns.")

        # Check for expected columns
        expected_columns = [
            "reviewId", "userName", "userImage", "content", "score", "thumbsUpCount",
            "reviewCreatedVersion", "at", "replyContent", "repliedAt", "sortOrder", "appId"
        ]
        for col in expected_columns:
            self.assertIn(col, df.columns, f"Missing expected column: {col}")

if __name__ == "__main__":
    unittest.main()