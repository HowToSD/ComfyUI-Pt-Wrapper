import os
import sys
import unittest
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)

from pytorch_wrapper.ptn_pre_add_channel_axis import PtnPreAddChannelAxis

class DummyModel(nn.Module):
    def forward(self, x):
        return x  # Pass-through model for testing


class TestPtnPreAddChannelAxis(unittest.TestCase):
    def setUp(self):
        """Set up test instance."""
        self.model_node = PtnPreAddChannelAxis()
        self.model = self.model_node.f(model=DummyModel())[0]
        

    def test_valid_shapes(self):
        """Test valid input shapes (rank 3 and 4)."""
        rank3_tensor = torch.randn(2, 8, 8)  # (bs, h, w)
        rank4_tensor = torch.randn(2, 3, 8, 8)  # (bs, c, h, w)

        result3 = self.model(rank3_tensor)
        result4 = self.model(rank4_tensor)

        self.assertEqual(result3.shape, (2, 1, 8, 8), f"Unexpected shape for rank 3: {result3.shape}")
        self.assertEqual(result4.shape, (2, 3, 8, 8), f"Unexpected shape for rank 4: {result4.shape}")

    def test_invalid_shapes(self):
        """Test invalid input shapes (rank 2 and rank 5)."""
        rank2_tensor = torch.randn(8, 8)  # (h, w)
        rank5_tensor = torch.randn(2, 3, 8, 8, 8)  # (bs, c, d, h, w)

        with self.assertRaises(ValueError, msg="Rank 2 tensor did not raise ValueError"):
            self.model(rank2_tensor)

        with self.assertRaises(ValueError, msg="Rank 5 tensor did not raise ValueError"):
            self.model(rank5_tensor)


if __name__ == "__main__":
    unittest.main()
