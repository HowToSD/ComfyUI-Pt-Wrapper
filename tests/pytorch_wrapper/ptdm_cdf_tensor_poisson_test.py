import os
import sys
import unittest
import torch
import scipy.stats as scst
import numpy as np

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)

from pytorch_wrapper.ptd_poisson import PoissonEx
from pytorch_wrapper.ptd_poisson import PtdPoisson
from pytorch_wrapper.ptdm_cdf_tensor import PtdmCdfTensor
from .utils import assert_equal_for_tensor_and_list_or_scalar

class TestPtdfCdfPoisson(unittest.TestCase):
    def setUp(self):
        """Set up test instance."""
        self.test_cases = {
            "Poisson": [
                # rate, x, expected
                (3, 2, scst.poisson.cdf(2, 3)),
                (3, 5, scst.poisson.cdf(5, 3)),
                (3, [2, 5], scst.poisson.cdf([2, 5], 3))  # 1d array for x
            ]
        }

    def test_custom_class_1(self):
        d = PoissonEx(rate=3)
        actual = d.cdf(torch.tensor(2)).numpy()
        expected = torch.tensor(0.42319008112684364).numpy()
        self.assertAlmostEqual(actual, expected, places=4, 
                                msg=f"Expected {expected}, got {actual}")

    def test_custom_class_2(self):
        d = PoissonEx(rate=3)
        actual = d.cdf(torch.tensor(5.0)).numpy()
        expected = torch.tensor(0.9160820579686966).numpy()
        self.assertAlmostEqual(actual, expected, places=4, 
                                msg=f"Expected {expected}, got {actual}")

    def test_custom_class_3(self):
        d = PoissonEx(rate=3)
        actual = d.cdf(torch.tensor([2.0, 5.0])).numpy()
        expected = torch.tensor([0.42319008112684364, 0.9160820579686966]).numpy()
        self.assertTrue(np.allclose(actual, expected, atol=1e-5), 
                                msg=f"Expected {expected}, got {actual}")

    def test_distributions(self):
        """Test probability mass and density functions for the target distribution."""
        for dist_name, cases in self.test_cases.items():  # For each distribution
            for case in cases:  # For each distribution's test cases
                with self.subTest(distribution=dist_name, params=case):
                    distribution = PtdPoisson().f(str(case[0]))[0]
                    cdf = PtdmCdfTensor().f(
                        distribution,
                        torch.tensor(case[1]))[0]

                    # Ensure return values are tensors
                    self.assertTrue(isinstance(cdf, torch.Tensor), "Probability output is not a tensor")

                    # Check values are equal
                    assert_equal_for_tensor_and_list_or_scalar(self, cdf, case[-1])


if __name__ == "__main__":
    unittest.main()
