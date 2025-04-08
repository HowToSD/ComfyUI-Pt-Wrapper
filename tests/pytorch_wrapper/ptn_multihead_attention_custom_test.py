import os
import sys
import unittest
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)

from pytorch_wrapper.ptn_multihead_attention_custom import PtnMultiheadAttentionCustom
from pytorch_wrapper.ptn_multihead_attention_custom import PtnMultiheadAttentionCustomDef

class TestPtnMultiheadAttentionCustom(unittest.TestCase):
    def setUp(self) -> None:
        """Set up test instance."""
        self.node = PtnMultiheadAttentionCustom()

    def test_fbatch_first(self) -> None:
        """Test forward pass with batch first."""
        model, = self.node.f(
            embed_dim=16,
            num_heads=4,
            dropout=0,
            bias=True,
            kdim=16,
            vdim=16,
            batch_first=True
        )

        self.assertIsInstance(model, PtnMultiheadAttentionCustomDef)

        # Create dummy input: batch_size=2, seq_len=5, embed_dim=16
        x = torch.rand(2, 5, 16)
        attention_custom_mask = torch.ones(2, 5)
        attn_output = model(x, attention_custom_mask)
        self.assertEqual(attn_output.shape, (2, 5, 16))

    def test_forward_seq_first(self) -> None:
        """Test forward pass with seq first."""
        model, = self.node.f(
            embed_dim=16,
            num_heads=4,
            dropout=0.1,
            bias=True,
            kdim=16,
            vdim=16,
            batch_first=False
        )

        self.assertIsInstance(model, PtnMultiheadAttentionCustomDef)

        # Create dummy input: batch_size=2, seq_len=5, embed_dim=16
        # Use seq first.
        x = torch.rand(5, 2, 16)
        attention_custom_mask = torch.ones(2, 5)
        attn_output = model(x, attention_custom_mask)
        self.assertEqual(attn_output.shape, (5, 2, 16))


if __name__ == "__main__":
    unittest.main()
