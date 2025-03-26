import os
import sys
import unittest
import torch

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)
from pytorch_wrapper.ptn_lstm_linear import PtnLSTMLinear


class TestPtnLSTMLinear(unittest.TestCase):
    def setUp(self):
        """Set up test instance."""
        self.model_node = PtnLSTMLinear()

    def test_seq_first_1_layer_uni(self):
        self.model = self.model_node.f(
            2, # input_size
            3, # hidden_size
            1, # num_layers
            True, # bias,
            False, # batch_first so [Seq, B, Token]
            0.0, # dropout
            False, # bidirectional,
            0,  # proj_size
            5, # linear output features
            True # linear bias
        )[0]

        x = torch.randn(10, 8, 2)  # [Seq, B, Token]
        outputs= self.model(x)
        self.assertEqual(outputs.size(), torch.Size([8, 5]))

    def test_seq_first_1_layer_bidi(self):
        self.model = self.model_node.f(
            2, # input_size
            3, # hidden_size
            1, # num_layers
            True, # bias,
            False, # batch_first so [Seq, B, Token]
            0.0, # dropout
            True, # bidirectional,
            0,  # proj_size
            5, # linear output features
            True # linear bias
        )[0]

        x = torch.randn(10, 8, 2)  # [Seq, B, Token]
        outputs= self.model(x)
        self.assertEqual(outputs.size(), torch.Size([8, 5]))

    def test_seq_first_2_layer_uni(self):
        self.model = self.model_node.f(
            2, # input_size
            3, # hidden_size
            2, # num_layers
            True, # bias,
            False, # batch_first so [Seq, B, Token]
            0.0, # dropout
            False, # bidirectional,
            0,  # proj_size
            5, # linear output features
            True # linear bias
        )[0]

        x = torch.randn(10, 8, 2)  # [Seq, B, Token]
        outputs= self.model(x)
        self.assertEqual(outputs.size(), torch.Size([8, 5]))

    def test_batch_first_1_layer_uni(self):
        self.model = self.model_node.f(
            2, # input_size
            3, # hidden_size
            1, # num_layers
            True, # bias,
            True, # batch_first
            0.0, # dropout
            False, # bidirectional,
            0,  # proj_size
            5, # linear output features
            True # linear bias
        )[0]

        x = torch.randn(8, 10, 2)  # [B, Seq, Token]
        outputs= self.model(x)
        self.assertEqual(outputs.size(), torch.Size([8, 5]))


if __name__ == "__main__":
    unittest.main()
