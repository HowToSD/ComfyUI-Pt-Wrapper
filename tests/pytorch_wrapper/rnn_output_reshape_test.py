import os
import sys
import unittest
import torch

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)

from pytorch_wrapper.rnn_output_reshape import extract_rnn_return_value_with_adjusted_label
from torch.nn import RNN

class TestExtractRnnReturnValueWithAdjustedLabel(unittest.TestCase):
    def run_test(self, num_layers, bidirectional, batch_first):
        input_size = 2
        hidden_size = 3
        batch_size = 8
        seq_len = 10

        model = RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity="tanh",
            bias=True,
            batch_first=batch_first,
            dropout=0.0,
            bidirectional=bidirectional
        )

        if batch_first:
            x = torch.randn(batch_size, seq_len, input_size)  # [B, Seq, Token]
        else:
            x = torch.randn(seq_len, batch_size, input_size)  # [Seq, B, Token]

        output_seq, _ = model(x)

        if batch_first:
            label = torch.randn(batch_size, seq_len, hidden_size)
        else:
            label = torch.randn(seq_len, batch_size, hidden_size)

        out, adjusted_label = extract_rnn_return_value_with_adjusted_label(
            x,
            label, output_seq, bidirectional, batch_first=batch_first
        )

        expected_hidden_dim = hidden_size * (2 if bidirectional else 1)
        self.assertEqual(out.shape, (batch_size, expected_hidden_dim))

        self.assertEqual(adjusted_label.shape, (batch_size, expected_hidden_dim))

        # y_hat and y
        self.assertEqual(out.shape, adjusted_label.shape)

    def test_uni_1_layer_seq_first(self):
        self.run_test(num_layers=1, bidirectional=False, batch_first=False)

    def test_uni_2_layer_seq_first(self):
        self.run_test(num_layers=2, bidirectional=False, batch_first=False)

    def test_bi_1_layer_seq_first(self):
        self.run_test(num_layers=1, bidirectional=True, batch_first=False)

    def test_bi_2_layer_seq_first(self):
        self.run_test(num_layers=2, bidirectional=True, batch_first=False)

    def test_uni_1_layer_batch_first(self):
        self.run_test(num_layers=1, bidirectional=False, batch_first=True)

    def test_uni_2_layer_batch_first(self):
        self.run_test(num_layers=2, bidirectional=False, batch_first=True)

    def test_bi_1_layer_batch_first(self):
        self.run_test(num_layers=1, bidirectional=True, batch_first=True)

    def test_bi_2_layer_batch_first(self):
        self.run_test(num_layers=2, bidirectional=True, batch_first=True)

    def test_valid_token_mean_batch_first(self):
        batch_size = 4
        seq_len = 6
        input_size = 5
        hidden_size = 7

        rnn = RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=False
        )

        # input: two valid tokens followed by padding (zero vectors)
        x = torch.cat([
            torch.randn(batch_size, 2, input_size),
            torch.zeros(batch_size, seq_len - 2, input_size)
        ], dim=1)

        label = torch.randn(batch_size, seq_len, hidden_size)
        output_seq, _ = rnn(x)

        out, adjusted_label = extract_rnn_return_value_with_adjusted_label(
            x,
            label,
            output_seq,
            bidirectional=False,
            batch_first=True,
            process_label=True,
            return_valid_token_mean=True
        )

        self.assertEqual(out.shape, (batch_size, hidden_size))
        self.assertEqual(adjusted_label.shape, (batch_size, hidden_size))

    def test_valid_token_mean_seq_first(self):
        batch_size = 4
        seq_len = 6
        input_size = 5
        hidden_size = 7

        rnn = RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=False,
            bidirectional=True
        )

        # input: three valid tokens followed by padding (zero vectors)
        x = torch.cat([
            torch.randn(3, batch_size, input_size),
            torch.zeros(seq_len - 3, batch_size, input_size)
        ], dim=0)

        label = torch.randn(seq_len, batch_size, hidden_size)
        output_seq, _ = rnn(x)

        out, adjusted_label = extract_rnn_return_value_with_adjusted_label(
            x,
            label,
            output_seq,
            bidirectional=True,
            batch_first=False,
            process_label=True,
            return_valid_token_mean=True
        )

        self.assertEqual(out.shape, (batch_size, hidden_size * 2))
        self.assertEqual(adjusted_label.shape, (batch_size, hidden_size * 2))


if __name__ == "__main__":
    unittest.main()
