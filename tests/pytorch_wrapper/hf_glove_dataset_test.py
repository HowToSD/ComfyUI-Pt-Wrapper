import os
import sys
import unittest
import torch

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)

from pytorch_wrapper.hf_glove_dataset import HfGloveDataset


class TestHfGloveDataset(unittest.TestCase):
    """
    Tests Hugging Face Dataset wrapper with word embedding support.
    """
    
    def setUp(self):
        """Set up test instance."""
        self.embedding_dim = 100
        self.max_seq_len = 256
        self.ds = HfGloveDataset(
            "imdb", # dataset_name
            "train", #split
            self.embedding_dim,
            self.max_seq_len
        )

    def test_vocabulary_size(self):
        self.assertEqual(400000, len(self.ds.glove))

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

    def test_loading(self):
        it = iter(self.ds)
        sample = next(it)
        
        # Compare text
        actual = sample[0].size()
        expected = torch.Size((self.max_seq_len, self.embedding_dim))
        self.assertEqual(expected, actual, f"expected {expected} and actual {actual} do not match.")

        # Compare label
        actual = sample[1]
        expected = torch.tensor(0, dtype=torch.int64)
        self.assertEqual(expected, actual, f"expected {expected} and actual {actual} do not match.")


if __name__ == "__main__":
    unittest.main()
