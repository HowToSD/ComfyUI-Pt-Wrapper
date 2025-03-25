import os
import sys
import unittest
import torch

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)

from pytorch_wrapper.ptv_hf_glove_dataset import PtvHfGloveDataset


class TestPtvHfGloveDataset(unittest.TestCase):
    """
    Tests the node for Hugging Face Dataset wrapper with word embedding support.
    """
    
    def setUp(self):
        """Set up test instance."""
        self.embedding_dim = 100
        self.max_seq_len = 256
        self.node = PtvHfGloveDataset()
        self.ds = self.node.f(
            "imdb", # dataset_name
            "train", #split
            self.embedding_dim,
            self.max_seq_len,
            "text",
            "label"
        )[0]

    def test_tokenize(self):
        """Test tokenization"""
        s = "Maddie is a cute puppy."
        actual = self.ds.tokenize(s)
        expected = ['maddie', 'is', 'a', 'cute', 'puppy']
        self.assertEqual(expected, actual, f"expected {expected} and actual {actual} do not match.")

    def test_vectorize(self):
        """Test embedding generation"""
        s = "Maddie is a cute puppy."
        tokens = self.ds.tokenize(s)
        embeddings = self.ds.vectorize(tokens)
        actual = embeddings.size()
        expected = torch.Size((self.max_seq_len, self.embedding_dim))
        self.assertEqual(expected, actual, f"expected {expected} and actual {actual} do not match.")


if __name__ == "__main__":
    unittest.main()
