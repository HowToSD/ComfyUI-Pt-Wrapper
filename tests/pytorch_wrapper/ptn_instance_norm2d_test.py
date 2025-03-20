import os
import sys
import unittest
import torch

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)
from pytorch_wrapper.ptn_instance_norm2d import PtnInstanceNorm2d


class TestPtnInstanceNorm2d(unittest.TestCase):
    def setUp(self):
        """Set up test instance."""
        self.model_node = PtnInstanceNorm2d()

    def check_normalization(self, x, num_features):
        model = self.model_node.f(num_features, True, True, 0.001)[0]
        out = model(x)
        self.assertEqual(out.shape, x.shape)
        
        # Compute std along normalized axes before and after
        std_before = x.std(dim=(-2,-1), unbiased=False)
        std_after = out.std(dim=(-2,-1), unbiased=False)
        
        mean_after = out.mean(dim=(-2,-1))

        # Ensure std is not one before normalization
        self.assertFalse(
            torch.allclose(
                std_before,
                torch.ones_like(std_after),
                atol=1e-3))

        # Ensure normalization happened (std should be 1 unless elements are very close)
        self.assertTrue(
            torch.allclose(
                std_after,
                torch.ones_like(std_after),
                atol=1e-3))
    
        self.assertTrue(
            torch.allclose(
                mean_after,
                torch.zeros_like(mean_after),
                atol=1e-3))

    def test_normalization_cases(self):
        test_cases = [
            # Data, num_features on the channel axis
            (torch.randn(3, 2, 2) * 2, 3),  # c = 3
            (torch.randn(1, 3, 2, 2) * 2, 3),  # c = 3
            (torch.randn(2, 3, 2, 2) * 2, 3),  # c = 3
            (torch.randn(8, 3, 2, 2) * 2, 3),  # c = 3
        ]
        
        for x, norm_shape in test_cases:
            with self.subTest(num_features=norm_shape):
                self.check_normalization(x, norm_shape)


if __name__ == "__main__":
    unittest.main()
