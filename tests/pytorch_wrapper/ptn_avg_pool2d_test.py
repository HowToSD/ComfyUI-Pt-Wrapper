import os
import sys
import unittest
import torch

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)
from pytorch_wrapper.ptn_avg_pool2d import PtnAvgPool2d


class TestPtnAvgPool2dModel(unittest.TestCase):
    def setUp(self):
        """Set up test instance."""
        self.model_node = PtnAvgPool2d()

    def test_kernel_stride_padding_as_int(self):
        self.model = self.model_node.f(
            "2",  # kernel_size (string)
            "2",  # stride (string)
            "0"  # padding (string)
        )[0]
        x = torch.ones(1, 3, 64, 64)
        out = self.model(x)
        self.assertEqual(out.size(), torch.Size([1, 3, 32, 32]))

    def test_kernel_stride_padding_as_tuple(self):
        self.model = self.model_node.f(
            "(2, 2)",  # kernel_size (tuple as string)
            "(2, 2)",  # stride (tuple as string)
            "(1, 1)"  # padding (tuple as string)
        )[0]
        x = torch.ones(1, 3, 64, 64)
        out = self.model(x)
        self.assertEqual(out.size(), torch.Size([1, 3, 33, 33]))

    def test_padding_as_different_values(self):
        for padding in ["0", "1"]:
            with self.subTest(padding=padding):
                self.model = self.model_node.f(
                    "2",  # kernel_size (string)
                    "2",  # stride (string)
                    padding  # padding (string)
                )[0]
                x = torch.ones(1, 3, 64, 64)
                out = self.model(x)
                expected_size = 32 + int(padding)  # Adjust for padding effect
                self.assertEqual(out.size(), torch.Size([1, 3, expected_size, expected_size]))


if __name__ == "__main__":
    unittest.main()
