import os
import sys
import unittest
import torch

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)
from pytorch_wrapper.ptn_embedding_rnn_linear import PtnEmbeddingRNNLinear


class TestPtnEmbeddingRNNLinear(unittest.TestCase):
    def setUp(self):
        """Set up test instance."""
        self.model_node = PtnEmbeddingRNNLinear()

    def test_1(self):
        self.model = self.model_node.f(
            32000, # vocabulary_size
            2, # input_size
            3, # hidden_size
            1, # num_layers
            "tanh", # nonlinearity
            True, # bias,
            True, # batch_first
            0.0, # dropout
            False, # bidirectional,
            5, # linear output features
            True # linear bias
        )[0]

        x = torch.randint(0, 32000, (8 * 5,)).to(torch.int64)
        x = x.view(8, 5)
        masks = torch.ones_like(x)
        outputs = self.model((x, masks))
        self.assertEqual(outputs.size(), torch.Size([8, 5]))



if __name__ == "__main__":
    unittest.main()
