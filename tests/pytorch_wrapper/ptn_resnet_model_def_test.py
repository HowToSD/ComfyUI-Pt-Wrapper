import os
import sys
import unittest
import torch
PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)
from pytorch_wrapper.ptn_resnet_model_def import ResnetModel, SingleResnetBlock, NResnetBlocks


class TestPtnResnetModelDef(unittest.TestCase):
    def setUp(self):
        """Setup method for initializing test conditions."""
        pass

    def test_single_resnet_block_no_downsample(self):
        """Test SingleResnetBlock without downsampling."""
        model = SingleResnetBlock(
                 in_channels=4,
                 out_channels=32,
                 kernel_size=3,
                 downsample=False)
        x = torch.ones(1, 4, 28, 28)
        out = model(x)
        self.assertEqual(out.size(), torch.Size([1, 32, 28, 28]))

    def test_single_resnet_block_downsample(self):
        """Test SingleResnetBlock with downsampling."""
        model = SingleResnetBlock(
                 in_channels=4,
                 out_channels=32,
                 kernel_size=3,
                 downsample=True)
        x = torch.ones(1, 4, 28, 28)
        out = model(x)
        self.assertEqual(out.size(), torch.Size([1, 32, 14, 14]))

    def test_n_resnet_blocks_no_downsample(self):
        """Test NResnetBlocks with multiple blocks and no downsampling."""
        model = NResnetBlocks(
                 num_blocks=3,
                 in_channels=4,
                 out_channels=32,
                 kernel_size=3,
                 downsample=False)
        x = torch.ones(1, 4, 28, 28)
        out = model(x)
        self.assertEqual(out.size(), torch.Size([1, 32, 28, 28]))

    def test_n_resnet_blocks_with_downsample(self):
        """Test NResnetBlocks with multiple blocks and downsampling."""
        model = NResnetBlocks(
                 num_blocks=3,
                 in_channels=4,
                 out_channels=32,
                 kernel_size=3,
                 downsample=True)
        x = torch.ones(1, 4, 28, 28)
        out = model(x)
        self.assertEqual(out.size(), torch.Size([1, 32, 14, 14]))

    def test_resnet_model_output_size(self):
        """Test ResnetModel output size."""
        model = ResnetModel(
                 num_blocks=3,
                 in_channels=4,
                 input_height=28,
                 input_width=28,
                 output_dim=8)
        x = torch.ones(1, 4, 28, 28)
        out = model(x)
        self.assertEqual(out.size(), torch.Size([1, 8]))

    def test_resnet_model_output_size2(self):
        """Test ResnetModel output size."""
        model = ResnetModel(
                 num_blocks=3,
                 in_channels=3,
                 input_height=256,
                 input_width=256,
                 output_dim=5)
        x = torch.ones(2, 3, 256, 256)
        out = model(x)
        self.assertEqual(out.size(), torch.Size([2, 5]))


if __name__ == "__main__":
    unittest.main()
