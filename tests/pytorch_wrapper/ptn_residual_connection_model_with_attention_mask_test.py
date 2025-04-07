import os
import sys
import unittest
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)

from pytorch_wrapper.ptn_residual_connection_model_with_attention_mask import PtnResidualConnectionModelWithAttentionMask
from pytorch_wrapper.ptn_multihead_attention import PtnMultiheadAttention

class TestPtnResidualConnectionModelWithAttentionMask(unittest.TestCase):
    def setUp(self) -> None:
        """Set up test instance with dummy model."""
        self.node = PtnResidualConnectionModelWithAttentionMask()
        self.inner_model = PtnMultiheadAttention().f(
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
        self.closure = nn.GELU()

    def test_forward_pass(self) -> None:
        """Test forward pass with residual addition and closure."""
        model, = self.node.f(self.inner_model, self.closure)
        x = torch.rand(2, 5, 16)
        attention_mask = torch.ones(2, 5)
        output = model(x, attention_mask)
        self.assertEqual(output.shape, (2, 5, 16))


if __name__ == "__main__":
    unittest.main()
