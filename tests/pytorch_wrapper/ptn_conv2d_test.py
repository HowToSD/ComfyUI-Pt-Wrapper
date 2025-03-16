import os
import sys
import unittest
import torch

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)
from pytorch_wrapper.ptn_conv2d import PtnConv2d


class TestPtnConv2d(unittest.TestCase):
    def setUp(self):
        """Set up test instance."""
        self.model_node = PtnConv2d()

    def test_kernel_stride_padding_as_int(self):
        self.model = self.model_node.f(
            3,  # in_channels
            32,  # out_channels
            "3",  # kernel_size (int)
            "1",  # stride (int)
            "1",  # padding (int)
            "1",  # dilation
            1,  # groups
            True,  # bias
            "zeros"  # padding_mode
        )[0]
        x = torch.ones(1, 3, 64, 64)
        out = self.model(x)
        self.assertEqual(out.size(), torch.Size([1, 32, 64, 64]))
        param_names = list(map(lambda e: e[0], self.model.named_parameters()))
        self.assertTrue("bias" in param_names)

    def test_kernel_stride_padding_as_tuple(self):
        self.model = self.model_node.f(
            3,  # in_channels
            1,  # out_channels
            "(3, 3)",  # kernel_size (tuple)
            "(1, 1)",  # stride (tuple)
            "(1, 1)",  # padding (tuple)
            "(1, 1)",  # dilation (tuple)
            1,  # groups
            False,  # bias
            "zeros"  # padding_mode
        )[0]
        x = torch.ones(1, 3, 64, 64)
        out = self.model(x)
        self.assertEqual(out.size(), torch.Size([1, 1, 64, 64]))
        param_names = list(map(lambda e: e[0], self.model.named_parameters()))
        self.assertFalse("bias" in param_names)

    def test_padding_as_same_valid(self):
        for padding_mode in ["zeros", "reflect", "replicate", "circular"]:
            with self.subTest(padding_mode=padding_mode):
                self.model = self.model_node.f(
                    3,  # in_channels
                    16,  # out_channels
                    "3",  # kernel_size (string)
                    "1",  # stride (string)
                    "same",  # padding (string "same")
                    "1",  # dilation (string)
                    1,  # groups
                    True,  # bias
                    padding_mode  # padding_mode
                )[0]
                x = torch.ones(1, 3, 64, 64)
                out = self.model(x)
                self.assertEqual(out.size(), torch.Size([1, 16, 64, 64]))
                
                self.model = self.model_node.f(
                    3,  # in_channels
                    16,  # out_channels
                    "3",  # kernel_size (string)
                    "1",  # stride (string)
                    "valid",  # padding (string "valid")
                    "1",  # dilation (string)
                    1,  # groups
                    True,  # bias
                    padding_mode  # padding_mode
                )[0]
                x = torch.ones(1, 3, 64, 64)
                out = self.model(x)
                self.assertEqual(out.size(), torch.Size([1, 16, 62, 62]))

    def test_groups_2(self):
        self.model = self.model_node.f(
            4,  # in_channels (divisible by groups)
            16,  # out_channels
            "3",  # kernel_size
            "1",  # stride
            "same",  # padding
            "1",  # dilation
            2,  # groups
            True,  # bias
            "zeros"  # padding_mode
        )[0]
        x = torch.ones(1, 4, 64, 64)
        out = self.model(x)
        self.assertEqual(out.size(), torch.Size([1, 16, 64, 64]))


if __name__ == "__main__":
    unittest.main()
