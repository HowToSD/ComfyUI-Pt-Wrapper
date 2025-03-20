import os
import sys
import unittest
import torch

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)
from pytorch_wrapper.ptn_layer_norm import PtnLayerNorm


class TestPtnLayerNorm(unittest.TestCase):
    def setUp(self):
        """Set up test instance."""
        self.model_node = PtnLayerNorm()

    def check_normalization(self, x, normalized_shape):
        model = self.model_node.f(str(normalized_shape), True, True)[0]
        out = model(x)
        self.assertEqual(out.shape, x.shape)
        
        # Compute std along normalized axes before and after
        std_before = x.std(dim=tuple(
            range(-len(normalized_shape), 0)
            ), unbiased=False)
        std_after = out.std(dim=tuple(
            range(-len(normalized_shape), 0)  # (-2, -1) for last two or (-3, -2, -1) for last three
            ), unbiased=False)
        
        mean_after = out.mean(dim=tuple(
            range(-len(normalized_shape), 0)  # (-2, -1) for last two or (-3, -2, -1) for last three
            ))

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
            # Data, normalized axis
            (torch.randn(1, 2, 4) * 2, [4]),  # token
            (torch.randn(1, 2, 4) * 2, [2, 4]),  # seq and token => per sample
            (torch.randn(2, 2, 4) * 2, [2, 4]),  # seq and token => per sample
            (torch.randn(1, 3, 2, 2) * 2, [2, 2]), # h, w
            (torch.randn(1, 3, 2, 2) * 2, [3, 2, 2]), # c, h, w => per sample
        ]
        
        for x, norm_shape in test_cases:
            with self.subTest(normalized_shape=norm_shape):
                self.check_normalization(x, norm_shape)


if __name__ == "__main__":
    unittest.main()
