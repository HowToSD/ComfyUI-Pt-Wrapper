import os
import sys
import unittest
import torch

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)
from pytorch_wrapper.ptn_lstm import PtnLSTM


class TestPtnLSTM(unittest.TestCase):
    def setUp(self):
        """Set up test instance."""
        self.model_node = PtnLSTM()

    def test_shape_1_layer(self):
        self.model = self.model_node.f(
            2, # input_size
            3, # hidden_size
            1, # num_layers
            True, # bias,
            False, # batch_first so [Seq, B, Token]
            0.0, # dropout
            False, # bidirectional,
            0  # proj_size
        )[0]

        x = torch.randn(10, 8, 2)  # [Seq, B, Token]
        out_seq_last_layer, (out_last_value, cn) = self.model(x)
        self.assertEqual(out_last_value.size(), torch.Size([1, 8, 3]))  # num_layers, batch, hidden_size
        self.assertEqual(out_seq_last_layer.size(), torch.Size([10, 8, 3]))  # seq, batch, hidden_size

    def test_shape_2_layers(self):
        self.model = self.model_node.f(
            2, # input_size
            3, # hidden_size
            2, # num_layers
            True, # bias,
            False, # batch_first so [Seq, B, Token]
            0.0, # dropout
            False, # bidirectional
            0  # proj_size
        )[0]

        x = torch.randn(10, 8, 2)  # [Seq, B, Token]
        out_seq_last_layer, (out_last_value, cn) = self.model(x)
        self.assertEqual(out_last_value.size(), torch.Size([2, 8, 3]))  # num_layers, batch, hidden_size
        self.assertEqual(out_seq_last_layer.size(), torch.Size([10, 8, 3]))  # seq, batch, hidden_size

    def test_bidirectional_1_layer(self):
        self.model = self.model_node.f(
            2, # input_size
            3, # hidden_size
            1, # num_layers
            True, # bias,
            False, # batch_first so [Seq, B, Token]
            0.0, # dropout
            True, # bidirectional
            0  # proj_size
        )[0]

        x = torch.randn(10, 8, 2)  # [Seq, B, Token]
        out_seq_last_layer, (out_last_value, cn) = self.model(x)
        self.assertEqual(out_last_value.size(), torch.Size([2, 8, 3]))  # 2 * num_layers, batch, hidden_size
        self.assertEqual(out_seq_last_layer.size(), torch.Size([10, 8, 6]))  # seq, batch, 2 * hidden_size

    def test_bidirectional_2_layers(self):
        self.model = self.model_node.f(
            2, # input_size
            3, # hidden_size
            2, # num_layers
            True, # bias,
            False, # batch_first so [Seq, B, Token]
            0.0, # dropout
            True, # bidirectional
            0  # proj_size
        )[0]

        x = torch.randn(10, 8, 2)  # [Seq, B, Token]
        out_seq_last_layer, (out_last_value, cn) = self.model(x)
        self.assertEqual(out_last_value.size(), torch.Size([4, 8, 3]))  # 2 * num_layers, batch, hidden_size
        self.assertEqual(out_seq_last_layer.size(), torch.Size([10, 8, 6]))  # seq, batch, 2 * hidden_size

    # batch-first "bf" test cases
    def test_bf_shape_1_layer(self):
        self.model = self.model_node.f(
            2, # input_size
            3, # hidden_size
            1, # num_layers
            True, # bias,
            True, # batch_first so [B, Seq, Token]
            0.0, # dropout
            False, # bidirectional
            0  # proj_size
        )[0]

        x = torch.randn(8, 10, 2)  # [B, Seq, Token]
        out_seq_last_layer, (out_last_value, cn) = self.model(x)
        self.assertEqual(out_last_value.size(), torch.Size([1, 8, 3]))  # num_layers, batch, hidden_size
        self.assertEqual(out_seq_last_layer.size(), torch.Size([8, 10, 3]))  # seq, batch, hidden_size

    def test_bf_shape_2_layers(self):
        self.model = self.model_node.f(
            2, # input_size
            3, # hidden_size
            2, # num_layers
            True, # bias,
            True, # batch_first so [B, Seq, Token]
            0.0, # dropout
            False, # bidirectional
            0  # proj_size
        )[0]

        x = torch.randn(8, 10, 2)  # [B, Seq, Token]
        out_seq_last_layer, (out_last_value, cn) = self.model(x)
        self.assertEqual(out_last_value.size(), torch.Size([2, 8, 3]))  # num_layers, batch, hidden_size
        self.assertEqual(out_seq_last_layer.size(), torch.Size([8, 10, 3]))  # batch, seq, hidden_size

    def test_bf_bidirectional_1_layer(self):
        self.model = self.model_node.f(
            2, # input_size
            3, # hidden_size
            1, # num_layers
            True, # bias,
            True, # batch_first so [B, Seq, Token]
            0.0, # dropout
            True, # bidirectional
            0  # proj_size
        )[0]

        x = torch.randn(8, 10, 2)  # [B, Seq, Token]
        out_seq_last_layer, (out_last_value, cn) = self.model(x)
        self.assertEqual(out_last_value.size(), torch.Size([2, 8, 3]))  # 2 * num_layers, batch, hidden_size
        self.assertEqual(out_seq_last_layer.size(), torch.Size([8, 10, 6]))  # batch, seq, 2 * hidden_size

    def test_bf_bidirectional_2_layers(self):
        self.model = self.model_node.f(
            2, # input_size
            3, # hidden_size
            2, # num_layers
            True, # bias,
            True, # batch_first so [B, Seq, Token]
            0.0, # dropout
            True, # bidirectional
            0  # proj_size
        )[0]

        x = torch.randn(8, 10, 2)  # [B, Seq, Token]
        out_seq_last_layer, (out_last_value, cn) = self.model(x)
        self.assertEqual(out_last_value.size(), torch.Size([4, 8, 3]))  # 2 * num_layers, batch, hidden_size
        self.assertEqual(out_seq_last_layer.size(), torch.Size([8, 10, 6]))  # batch, seq, 2 * hidden_size


if __name__ == "__main__":
    unittest.main()
