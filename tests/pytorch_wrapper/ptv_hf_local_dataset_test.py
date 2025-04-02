import os
import sys
import unittest
import torch

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)

from hugging_face_wrapper.hf_tokenizer_encode import HfTokenizerEncode
from pytorch_wrapper.ptv_hf_local_dataset import PtvHfLocalDataset


class TestPtvHfLocalDataset(unittest.TestCase):
    """
    Tests the node for Hugging Face Dataset wrapper with word embedding support.
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

        self.ds = PtvHfLocalDataset().f(
            "imdb/train.jsonl", # dataset file path
            "json",
            sample_field_name = "text",
            label_field_name = "label",
            encode=encode,
            encode_return_dict=True,
            remove_html_tags=True
        )[0]

    def test_loading(self):

        # Check number of examples
        self.assertEquals(25000, len(self.ds))

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
