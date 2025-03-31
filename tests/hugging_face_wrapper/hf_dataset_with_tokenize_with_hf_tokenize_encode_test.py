import os
import sys
import unittest
import torch

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)

from hugging_face_wrapper.hf_dataset_with_token_encode import HfDatasetWithTokenEncode
from hugging_face_wrapper.hf_tokenizer_encode import HfTokenizerEncode


class TestHfDatasetWithTokenEncode(unittest.TestCase):
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

        self.ds = HfDatasetWithTokenEncode(
            "imdb", # dataset_name
            "train", #split
            encode=encode,
            encode_return_dict=True
        )
        
        self.ds_remove_html = HfDatasetWithTokenEncode(
            "imdb", # dataset_name
            "train", #split
            encode=encode,
            remove_html_tags=True,
            encode_return_dict=True
        )

    def test_loading(self):
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


    def test_loading_remove_html(self):
        it = iter(self.ds_remove_html)
        (sample, mask), label = next(it)

        # Compare text
        actual = sample.size()
        expected = torch.Size((self.max_length, ))
        self.assertEqual(expected, actual, f"expected {expected} and actual {actual} do not match.")

        # Compare mask
        actual = mask.size()
        expected = torch.Size((self.max_length, ))
        self.assertEqual(expected, actual, f"expected {expected} and actual {actual} do not match.")

        # Compare label
        actual = label
        expected = torch.tensor(0, dtype=torch.int64)
        self.assertEqual(expected, actual, f"expected {expected} and actual {actual} do not match.")


if __name__ == "__main__":
    unittest.main()
