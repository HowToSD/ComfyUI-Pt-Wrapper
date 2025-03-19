import os
import sys
import unittest
import torch
import scipy.stats as scst
import numpy as np

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)

from pytorch_wrapper.ptd_beta import BetaEx
from pytorch_wrapper.ptd_beta import PtdBeta
from pytorch_wrapper.ptdm_icdf_tensor import PtdmIcdfTensor
from .utils import assert_equal_for_tensor_and_list_or_scalar

class TestPtdfIcdfBeta(unittest.TestCase):
    def setUp(self):
        """Set up test instance."""
        self.test_cases = {
            "Beta": [
                # str alpha, str beta, num q, expected
                ("2", "8", 0.5, scst.beta.ppf(0.5, 2, 8)),
                ("2", "8", 0, scst.beta.ppf(0, 2, 8)),
                ("2", "8", [0, 0.5, 1], scst.beta.ppf([0, 0.5, 1], 2, 8))  # 1d array for q
            ]
        }

    def test_custom_class_1(self):
        d = BetaEx(concentration1=2, concentration0=8)
        actual = d.icdf(torch.tensor(0.98046875)).numpy()
        expected = torch.tensor(0.5).numpy()
        self.assertAlmostEqual(actual, expected, places=4, 
                                msg=f"Expected {expected}, got {actual}")

    def test_custom_class_2(self):
        d = BetaEx(concentration1=2, concentration0=8)
        actual = d.icdf(torch.tensor(0.0)).numpy()
        expected = torch.tensor(0.0).numpy()
        self.assertAlmostEqual(actual, expected, places=4, 
                                msg=f"Expected {expected}, got {actual}")

    def test_custom_class_3(self):
        d = BetaEx(concentration1=2, concentration0=8)
        actual = d.icdf(torch.tensor(1.0)).numpy()
        expected = torch.tensor(1.0).numpy()
        self.assertAlmostEqual(actual, expected, places=4, 
                                msg=f"Expected {expected}, got {actual}")


    def test_distributions(self):
        """Test probability mass and density functions for the target distribution."""
        for dist_name, cases in self.test_cases.items():  # For each distribution
            for case in cases:  # For each distribution's test cases
                with self.subTest(distribution=dist_name, params=case):
                    distribution = PtdBeta().f(
                        alpha=case[0], 
                        beta=case[1])[0]
                    icdf = PtdmIcdfTensor().f(
                        distribution,
                        torch.tensor(case[2]))[0]

                    # Ensure return values are tensors
                    self.assertTrue(isinstance(icdf, torch.Tensor), "Output is not a tensor")

                    # Check values are equal
                    assert_equal_for_tensor_and_list_or_scalar(self, icdf, case[-1])


if __name__ == "__main__":
    unittest.main()
