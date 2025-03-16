import os
import sys
import unittest
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)

from pytorch_wrapper.ptn_pre_flatten import PtnPreFlatten

class DummyModel(nn.Module):
    def forward(self, x):
        return x  # Pass-through model for testing


class TestPtnPreFlatten(unittest.TestCase):
    def setUp(self):
        """Set up test instance."""
        self.model_node = PtnPreFlatten()
        self.model = self.model_node.f(model=DummyModel())[0]

    def test_valid_shapes(self):
        """Test flattening on valid input shapes."""
        rank3_tensor = torch.randn(2, 8, 8)  # (bs, h, w)
        rank4_tensor = torch.randn(2, 3, 8, 8)  # (bs, c, h, w)
        rank5_tensor = torch.randn(2, 3, 4, 8, 8)  # (bs, c, d, h, w)

        result3 = self.model(rank3_tensor)
        result4 = self.model(rank4_tensor)
        result5 = self.model(rank5_tensor)

        self.assertEqual(result3.shape, (2, 64), f"Unexpected shape for rank 3: {result3.shape}")
        self.assertEqual(result4.shape, (2, 192), f"Unexpected shape for rank 4: {result4.shape}")
        self.assertEqual(result5.shape, (2, 768), f"Unexpected shape for rank 5: {result5.shape}")


if __name__ == "__main__":
    unittest.main()
