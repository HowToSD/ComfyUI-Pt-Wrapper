import os
import sys
import unittest
import torch
from itertools import product

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)

from pytorch_wrapper.utils import pad_truncate_sequence


class TestPadTruncateSequence(unittest.TestCase):
    def setUp(self):
        # 3 sample sequences of different lengths
        self.token_tensor_list = [
            torch.tensor([1, 2, 3], dtype=torch.long),
            torch.tensor([4, 5], dtype=torch.long),
            torch.tensor([6], dtype=torch.long)
        ]
        self.max_length = 255

    def run_case(self, padding, padding_method, padding_value, truncation):

        with self.subTest(
            padding=padding,
            padding_method=padding_method,
            padding_value=padding_value,
            truncation=truncation
        ):
            tokens, masks = pad_truncate_sequence(
                token_tensor_list=self.token_tensor_list,
                padding=padding,
                padding_method=padding_method,
                padding_value=padding_value,
                truncation=truncation,
                max_length=self.max_length
            )

            if padding:
                # Tokens and masks should be stacked tensors
                self.assertIsInstance(tokens, torch.Tensor)
                self.assertIsInstance(masks, torch.Tensor)
                self.assertEqual(tokens.shape, masks.shape)
                self.assertEqual(tokens.dim(), 2)  # batch x seq_len
                self.assertEqual(tokens.size(0), 3)
                if padding_method == "max_length":
                    self.assertEqual(tokens.size(1), self.max_length)
                elif padding_method == "longest":
                    truncated_lengths = [min(len(t), self.max_length) if truncation else len(t)
                                         for t in self.token_tensor_list]
                    expected_len = max(truncated_lengths)
                    self.assertEqual(tokens.size(1), expected_len)
            else:
                # Should return lists
                self.assertIsInstance(tokens, list)
                self.assertIsInstance(masks, list)
                self.assertEqual(len(tokens), 3)
                for t, m in zip(tokens, masks):
                    self.assertEqual(t.size(), m.size())
                    expected_len = min(len(t), self.max_length) if truncation else len(t)
                    self.assertEqual(t.size(0), expected_len)

    def test_all_cases(self):
        for padding, padding_method, padding_value, truncation in product(
            [True, False],
            ["max_length", "longest"],
            [0, 255],
            [True, False]
        ):
            self.run_case(padding, padding_method, padding_value, truncation)

    def test_single_tensor_input(self):
        """Test behavior when token_tensor_list has only one element."""
        single_tensor_list = [torch.tensor([1, 2, 3, 4], dtype=torch.long)]

        for padding, padding_method, padding_value, truncation in product(
            [True, False],
            ["max_length", "longest"],
            [0, 255],
            [True, False]
        ):
            with self.subTest(
                single_input=True,
                padding=padding,
                padding_method=padding_method,
                padding_value=padding_value,
                truncation=truncation
            ):
                tokens, masks = pad_truncate_sequence(
                    token_tensor_list=single_tensor_list,
                    padding=padding,
                    padding_method=padding_method,
                    padding_value=padding_value,
                    truncation=truncation,
                    max_length=self.max_length
                )

                # Always stacked even if padding is False
                self.assertIsInstance(tokens, torch.Tensor)
                self.assertIsInstance(masks, torch.Tensor)
                self.assertEqual(tokens.shape, masks.shape)
                self.assertEqual(tokens.size(0), 1)

                expected_len = 4
                if truncation:
                    expected_len = min(expected_len, self.max_length)
                if padding:
                    if padding_method == "max_length":
                        expected_len = self.max_length
                    elif padding_method == "longest":
                        expected_len = expected_len  # no change

                self.assertEqual(tokens.size(1), expected_len)



if __name__ == "__main__":
    unittest.main()
