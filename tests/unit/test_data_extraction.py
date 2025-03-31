import os
import sys
import unittest

import pandas as pd

from src.data_extraction import load_csv_file


class TestDataExtraction(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Define the dataset path before running tests"""

        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(
            os.path.join(current_dir, "..", "..")
        )

        cls.dataset_path = os.path.join(project_root, "dataset.csv")
        sys.path.insert(0, project_root)

        if not os.path.exists(cls.dataset_path):
            raise FileNotFoundError(
                f"Dataset file not found: {cls.dataset_path}"
            )

    def test_load_real_dataset(self):
        """Test if the dataset loads correctly as a DataFrame."""
        df = load_csv_file(self.dataset_path)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(df.shape[0], 0, "Dataset is empty (no rows).")
        self.assertGreater(df.shape[1], 0, "Dataset has no columns.")

        expected_columns = [
            "reviewId", "userName", "userImage", "content",
            "score", "thumbsUpCount", "reviewCreatedVersion",
            "at", "replyContent", "repliedAt",
            "sortOrder", "appId"
        ]

        for col in expected_columns:
            self.assertIn(col, df.columns, f"Missing expected column: {col}")


if __name__ == "__main__":
    unittest.main()
