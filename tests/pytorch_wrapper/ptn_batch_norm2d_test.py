import os
import sys
import unittest
import torch

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)
from pytorch_wrapper.ptn_batch_norm2d import PtnBatchNorm2d


class TestPtnBatchNorm2d(unittest.TestCase):
    def setUp(self):
        """Set up test instance."""
        self.model_node = PtnBatchNorm2d()

    def check_normalization(self, x, num_features):
        model = self.model_node.f(num_features,
                                  affine=False,
                                  track_running_stats=False,
                                  momentum=0.001)[0]
        
        out = model(x)
        self.assertEqual(out.shape, x.shape)

        # Compute std along normalized axes before and after
        std_before = x.std(dim=(0, 2, 3), unbiased=True)
        std_after = out.std(dim=(0, 2, 3), unbiased=True)
        mean_after = out.mean(dim=(0, 2, 3))

        # Ensure std is not one before normalization
        self.assertFalse(
            torch.allclose(
                std_before,
                torch.ones_like(std_after),
                atol=1e-1))

        # Ensure normalization happened (std should be 1 unless elements are very close)
        self.assertTrue(
            torch.allclose(
                std_after,
                torch.ones_like(std_after),
                atol=1e-1))
    
        self.assertTrue(
            torch.allclose(
                mean_after,
                torch.zeros_like(mean_after),
                atol=1e-1))

    def test_normalization_cases(self):
        test_cases = [
            # Data, num_features on the channel axis
            (torch.randn(2, 3, 3, 3) * 2, 3),  # c = 3
            (torch.randn(3, 3, 3, 3) * 2, 3),  # c = 3
        ]
        
        for x, norm_shape in test_cases:
            with self.subTest(num_features=norm_shape):
                self.check_normalization(x, norm_shape)


if __name__ == "__main__":
    unittest.main()
