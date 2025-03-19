import os
import sys
import unittest
import torch
import scipy.stats as scst


PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)

from pytorch_wrapper.ptd_chi2 import Chi2Ex
from pytorch_wrapper.ptd_chi2 import PtdChi2
from pytorch_wrapper.ptdm_icdf_tensor import PtdmIcdfTensor
from .utils import assert_equal_for_tensor_and_list_or_scalar

class TestPtdfIcdfChi2(unittest.TestCase):
    def setUp(self):
        """Set up test instance."""
        self.test_cases = {
            "Chi2": [
                # str df, num q, expected
                ("40.0", 0.5, scst.chi2.ppf(0.5, 40)),
                ("40.0", 0.9, scst.chi2.ppf(0.9, 40)),
                ("40.0", [0.8, 0.9], scst.chi2.ppf([0.8, 0.9], 40))  # 1d array for q
            ]
        }

    def test_custom_class_1(self):
        d = Chi2Ex(df=1)
        actual = d.icdf(torch.tensor(0.01)).numpy()
        expected = torch.tensor(0.00015708785790970184).numpy()
        self.assertAlmostEqual(actual, expected, places=4, 
                                msg=f"Expected {expected}, got {actual}")

    def test_custom_class_2(self):
        d = Chi2Ex(df=24)
        actual = d.icdf(torch.tensor(0.6)).numpy()
        expected = torch.tensor(25.10634821892835).numpy()
        self.assertAlmostEqual(actual, expected, places=4, 
                                msg=f"Expected {expected}, got {actual}")


    def test_distributions(self):
        """Test probability mass and density functions for the target distribution."""
        for dist_name, cases in self.test_cases.items():  # For each distribution
            for case in cases:  # For each distribution's test cases
                with self.subTest(distribution=dist_name, params=case):
                    distribution = PtdChi2().f(
                        df=case[0])[0]
                    icdf = PtdmIcdfTensor().f(
                        distribution,
                        torch.tensor(case[1]))[0]

                    # Ensure return values are tensors
                    self.assertTrue(isinstance(icdf, torch.Tensor), "Output is not a tensor")

                    # Check values are equal
                    assert_equal_for_tensor_and_list_or_scalar(self, icdf, case[-1])


if __name__ == "__main__":
    unittest.main()
