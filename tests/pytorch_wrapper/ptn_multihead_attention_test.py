import os
import sys
import unittest
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)

from pytorch_wrapper.ptn_multihead_attention import PtnMultiheadAttention
from pytorch_wrapper.ptn_multihead_attention import PtnMultiheadAttentionDef

class TestPtnMultiheadAttention(unittest.TestCase):
    def setUp(self) -> None:
        """Set up test instance."""
        self.node = PtnMultiheadAttention()

    def test_forward_pass(self) -> None:
        """Test forward pass of MultiheadAttention with valid input."""
        model, = self.node.f(
            embed_dim=16,
            num_heads=4,
            dropout=0.1,
            bias=True,
            add_bias_kv=False,
            add_zero_attn=False,
            kdim=16,
            vdim=16,
            batch_first=True
        )

        self.assertIsInstance(model, PtnMultiheadAttentionDef)

        # Create dummy input: batch_size=2, seq_len=5, embed_dim=16
        x = torch.rand(2, 5, 16)
        attention_mask = torch.ones(2, 5)
        attn_output = model(x, attention_mask)
        self.assertEqual(attn_output.shape, (2, 5, 16))


    def test_forward_pass_with_zero_attn(self) -> None:
        """Test forward pass with add_zero_attn enabled."""
        model, = self.node.f(
            embed_dim=16,
            num_heads=4,
            dropout=0.0,
            bias=True,
            add_bias_kv=False,
            add_zero_attn=True,
            kdim=16,
            vdim=16,
            batch_first=True
        )

        x = torch.rand(2, 5, 16)
        attention_mask = torch.ones(2, 5)
        attn_output = model(x, attention_mask)

        self.assertEqual(attn_output.shape, (2, 5, 16))


    def test_forward_pass_with_bias_kv(self) -> None:
        """Test forward pass with bias_kv enabled."""
        model, = self.node.f(
            embed_dim=16,
            num_heads=4,
            dropout=0.0,
            bias=True,
            add_bias_kv=True,
            add_zero_attn=False,
            kdim=16,
            vdim=16,
            batch_first=True
        )

        x = torch.rand(2, 5, 16)
        attention_mask = torch.ones(2, 5)
        attn_output = model(x, attention_mask)

        self.assertEqual(attn_output.shape, (2, 5, 16))

if __name__ == "__main__":
    unittest.main()
