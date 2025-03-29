import os
import sys
import unittest
import torch

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)
from pytorch_wrapper.ptn_embedding_transformer_linear import PtnEmbeddingTransformerLinear


class TestPtnEmbeddingTransformerLinear(unittest.TestCase):
    def setUp(self):
        """Set up test instance."""
        self.model_node = PtnEmbeddingTransformerLinear()

    def test_1(self):
        self.model = self.model_node.f(
            1,  # num_encoder
            32000,  # vocabulary_size
            32, #  hidden_size
            8,  # nhead
            64, # dim_feedforward
            0.1, #  dropout
            "gelu", #  activation
            1e-5, # layer_norm_eps,
            True, #  batch_first
            False, # norm_first
            True, # bias
            512, #  max_length of seq
            1,  # linear_output_size
            True  # linear_bias
        )[0]

        x = torch.randint(0, 32000, (8 * 77,)).to(torch.int64)
        x = x.view(8, 77)
        masks = torch.ones_like(x)
        outputs = self.model((x, masks))
        self.assertEqual(outputs.size(), torch.Size([8, 1]))


if __name__ == "__main__":
    unittest.main()
