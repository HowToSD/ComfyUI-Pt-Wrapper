import os
import sys
import unittest
import torch

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)
from pytorch_wrapper.ptn_resnet_model import PtnResnetModel


class TestPtnResnetModel(unittest.TestCase):
    def setUp(self):
        """Set up test instance."""
        self.model_node = PtnResnetModel()

    def test_forward(self):
        model = self.model_node.f(
            input_dim="(4, 28, 28)",
            output_dim=10,
            num_blocks=2
        )[0]
        bs = 2
        x = torch.ones(bs, 4, 28, 28)
        out = model(x)
        self.assertEqual(out.size(), torch.Size([bs, 10]))

    def test_forward2(self):
        model = self.model_node.f(
            input_dim="(3, 256, 256)",
            output_dim=4,
            num_blocks=2
        )[0]
        bs = 4
        x = torch.ones(bs, 3, 256, 256)
        out = model(x)
        self.assertEqual(out.size(), torch.Size([bs, 4]))

    def test_invalid_input_shape(self):
        model = self.model_node.f(
            input_dim="(3, 32, 32)",
            output_dim=10,
            num_blocks=2
        )[0]
        bs = 2
        x = torch.ones(bs, 32, 32)  # Missing channel dimension
        with self.assertRaises(RuntimeError) as context:
            model(x)

    def test_invalid_output_dim(self):
        with self.assertRaises(ValueError):
            self.model_node.f(
                input_dim="(3, 28, 28)",
                output_dim=0,  # Invalid output_dim
                num_blocks=2
            )

    def test_invalid_num_blocks(self):
        with self.assertRaises(ValueError):
            self.model_node.f(
                input_dim="(3, 28, 28)",
                output_dim=10,
                num_blocks=0  # Invalid num_blocks
            )

if __name__ == "__main__":
    unittest.main()
