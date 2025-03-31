import os
import sys
import unittest
import torch

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)

from hugging_face_wrapper.hf_dataset_with_token_encode import HfDatasetWithTokenEncode
from sentencepiece_wrapper.sp_load_model import SpLoadModel
from sentencepiece_wrapper.sp_encode import SpEncode


class TestHfDatasetWithTokenEncode(unittest.TestCase):
    """
    Tests Hugging Face Dataset wrapper with custom token_encode function.
    """
    
    def setUp(self):
        """Set up test instance."""
        self.load_node = SpLoadModel()
        self.sp = self.load_node.f("spiece.model")[0]
        self.encode_node = SpEncode()
        self.max_length=15
        encode = self.encode_node.f(
            spmodel=self.sp,
            padding=False,
            padding_method="longest",
            padding_value=0,
            truncation=True,
            max_length=self.max_length
        )[0]

        self.ds = HfDatasetWithTokenEncode(
            "imdb", # dataset_name
            "train", #split
            encode=encode,
        )
        
        self.ds_remove_html = HfDatasetWithTokenEncode(
            "imdb", # dataset_name
            "train", #split
            encode=encode,
            remove_html_tags=True
        )

    def test_loading(self):
        it = iter(self.ds)
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
