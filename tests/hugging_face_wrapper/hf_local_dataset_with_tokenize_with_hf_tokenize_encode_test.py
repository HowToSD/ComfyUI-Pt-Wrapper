import os
import sys
import unittest
import torch

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)

from hugging_face_wrapper.hf_local_dataset import HfLocalDataset
from hugging_face_wrapper.hf_tokenizer_encode import HfTokenizerEncode


class TestHfLocalDataset(unittest.TestCase):
    """
    Tests Hugging Face Dataset wrapper with custom token_encode function.
    """
    
    def setUp(self):
        """Set up test instance."""
        self.max_length=15
        encode = HfTokenizerEncode().f(
            model_name="distilbert-base-uncased",
            padding=True,
            padding_method="longest",
            truncation=True,
            max_length=self.max_length
        )[0]

        self.ds = HfLocalDataset(
            "imdb/train.jsonl", # dataset file path
            "json",
            encode=encode,
            encode_return_dict=True
        )


    def test_loading(self):

        # Check number of examples
        self.assertEqual(25000, len(self.ds))

        it = iter(self.ds)
        (sample, mask), label = next(it)

        # Compare text
        actual = sample.size()
        expected = torch.Size((self.max_length,))
        self.assertEqual(expected, actual, f"expected {expected} and actual {actual} do not match.")

        # Compare mask
        actual = mask.size()
        expected = torch.Size((self.max_length,))
        self.assertEqual(expected, actual, f"expected {expected} and actual {actual} do not match.")

        # Compare label
        actual = label
        expected = torch.tensor(0, dtype=torch.int64)
        self.assertEqual(expected, actual, f"expected {expected} and actual {actual} do not match.")


if __name__ == "__main__":
    unittest.main()
