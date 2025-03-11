import os
import sys
import unittest
import torch
from PIL import Image
import numpy as np

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)
from pytorch_wrapper.ptv_transforms_data_augment import PtvTransformsDataAugment


class TestPtvTransformsResize(unittest.TestCase):
    def setUp(self):
        """Set up test data."""
        self.compose_node = PtvTransformsDataAugment()
        self.h_flip_prob = 0.4
        self.v_flip_prob = 0.6
        self.rotate_degree = 10
        self.h_translate_ratio = 0.1
        self.v_translate_ratio = 0.15
        self.min_scale = 0.9
        self.max_scale = 1.1

    def test_with_transform(self):
        """Tests the transformation output."""
        transform = self.compose_node.f(
            self.h_flip_prob,
            self.v_flip_prob,
            self.rotate_degree,
            self.h_translate_ratio,
            self.v_translate_ratio,
            self.min_scale,
            self.max_scale
        )[0]
        self.assertTrue(callable(transform), "Transform should be callable.")

        # Create a dummy image
        img = Image.fromarray(np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8))
        transformed_img = transform(img)

        # Check output type
        self.assertIsInstance(transformed_img, torch.Tensor, "Output should be a PyTorch tensor.")

        # Check output shape
        self.assertEqual(transformed_img.shape, (3, img.height, img.width), "Output shape should match target dimensions.")


if __name__ == "__main__":
    unittest.main()
