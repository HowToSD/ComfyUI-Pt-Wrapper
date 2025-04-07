import os
import sys
import unittest
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)

from pytorch_wrapper.ptn_chained_model_with_attention_mask import PtnChainedModelWithAttentionMask
from pytorch_wrapper.ptn_multihead_attention import PtnMultiheadAttention

class TestPtnChainedModel(unittest.TestCase):
    def setUp(self) -> None:
        """Set up test instance with dummy models."""
        self.node = PtnChainedModelWithAttentionMask()

        self.model_a = PtnMultiheadAttention().f(
            embed_dim=16,
            num_heads=4,
            dropout=0.1,
            bias=True,
            add_bias_kv=False,
            add_zero_attn=False,
            kdim=16,
            vdim=16,
            batch_first=True
        )[0]
        self.model_b = nn.LayerNorm([16])  # Normalize within each token
        self.closure = nn.ReLU()  # Apply ReLU as the closure

    def test_forward_pass(self) -> None:
        """Test forward pass through the chained model."""
        model, = self.node.f(
            self.model_a, 
            self.model_b,
            True, # a takes mask
            False, # b does not take mask 
            self.closure)


        x = torch.rand(2, 5, 16)
        attention_mask = torch.ones(2, 5)
        output = model(x, attention_mask)
        self.assertEqual(output.shape, (2, 5, 16))


if __name__ == "__main__":
    unittest.main()
