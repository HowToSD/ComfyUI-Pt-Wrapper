import os
import sys
import unittest
import torch
import torch.nn as nn
from typing import Any

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)
from pytorch_wrapper.ptn_embedding import PtnEmbedding


class TestPtnEmbedding(unittest.TestCase):
    def setUp(self) -> None:
        """Set up the test instance of PtnEmbedding."""
        self.model_node = PtnEmbedding()

    def test_basic_embedding(self) -> None:
        """Test that PtnEmbedding outputs expected shapes and has correct weight dimensions."""
        # Create embedding module
        model: nn.Module = self.model_node.f(
            100,  # vocabulary size
            8,    # embedding dimension
            -1    # no padding index
        )[0]

        # Check embedding weight shape
        self.assertEqual(model.weight.size(), torch.Size([100, 8]))

        # Create sample input: batch size 1, sequence length 6
        x: torch.Tensor = torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.int64)

        # Forward pass
        out: torch.Tensor = model(x)

        # Output shape should be (batch_size, sequence_length, embedding_dim)
        self.assertEqual(out.size(), torch.Size([1, 6, 8]))


    def test_padding_index(self) -> None:
        """Test that PtnEmbedding sets the correct padding_idx internally."""
        padding_idx: int = 2
        model: nn.Module = self.model_node.f(
            100,  # vocabulary size
            8,    # embedding dimension
            padding_idx
        )[0]

        # Ensure the model has the padding_idx attribute set correctly
        self.assertTrue(hasattr(model, "padding_idx"))
        self.assertEqual(model.padding_idx, padding_idx)

        # Check that the weights at padding_idx are all zeros initially
        self.assertTrue(torch.all(model.weight[padding_idx] == 0))



if __name__ == "__main__":
    unittest.main()

