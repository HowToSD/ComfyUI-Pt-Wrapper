import os
import sys
import unittest
import torch
import scipy.stats as scst
import numpy as np

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)

from pytorch_wrapper.ptd_student_t import StudentTEx
from pytorch_wrapper.ptd_student_t import PtdStudentT
from pytorch_wrapper.ptdm_cdf_tensor import PtdmCdfTensor
from .utils import assert_equal_for_tensor_and_list_or_scalar

class TestPtdfCdfStudentT(unittest.TestCase):
    def setUp(self):
        """Set up test instance."""
        self.test_cases = {
            "StudentT": [
                # str df, str loc, str scale, num x, expected
                ("40.0", "0", "1", 0.5, scst.t.cdf(0.5, 40)),
                ("40.0", "170", "20", 185, scst.t.cdf((185-170)/20, 40)), # adjust loc and scale
                ("40.0", "170", "20", [160.0, 185], scst.t.cdf([(160-170)/20, (185-170)/20], 40))  # 1d array for x
            ]
        }

    def test_custom_class_1(self):
        d = StudentTEx(df=40, loc=0, scale=1)
        actual = d.cdf(torch.tensor(0.5)).numpy()
        expected = torch.tensor(0.690092632383276).numpy()
        self.assertAlmostEqual(actual, expected, places=4, 
                                msg=f"Expected {expected}, got {actual}")

    def test_custom_class_2(self):
        d = StudentTEx(df=40, loc=0, scale=1)
        actual = d.cdf(torch.tensor(0.9)).numpy()
        expected = torch.tensor(0.813246561322997).numpy()
        self.assertAlmostEqual(actual, expected, places=4, 
                                msg=f"Expected {expected}, got {actual}")

    def test_custom_class_3(self):
        d = StudentTEx(df=40, loc=0, scale=1)
        actual = d.cdf(torch.tensor([0.5, 0.9])).numpy()
        expected = torch.tensor([0.690092632383276, 0.813246561322997]).numpy()
        self.assertTrue(np.allclose(actual, expected, atol=1e-5), 
                                msg=f"Expected {expected}, got {actual}")

    def test_distributions(self):
        """Test probability mass and density functions for the target distribution."""
        for dist_name, cases in self.test_cases.items():  # For each distribution
            for case in cases:  # For each distribution's test cases
                with self.subTest(distribution=dist_name, params=case):
                    distribution = PtdStudentT().f(
                        df=case[0], 
                        loc=case[1],
                        scale=case[2])[0]
                    cdf = PtdmCdfTensor().f(
                        distribution,
                        torch.tensor(case[3]))[0]

                    # Ensure return values are tensors
                    self.assertTrue(isinstance(cdf, torch.Tensor), "Probability output is not a tensor")

                    # Check values are equal
                    assert_equal_for_tensor_and_list_or_scalar(self, cdf, case[-1])


if __name__ == "__main__":
    unittest.main()
