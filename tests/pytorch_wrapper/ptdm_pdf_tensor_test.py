import os
import sys
import unittest
import numpy as np
import torch
import scipy.stats as scst
from torch.distributions import Exponential

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)

from pytorch_wrapper.ptdm_pdf_tensor import PtdmPdfTensor


class TestPtdfPdfTensor(unittest.TestCase):
    def setUp(self):
        """Set up test instance."""
        self.test_cases = {
            "Exponential": [
                (1.0, 3.0, scst.expon.pdf(3.0, scale=1)),
                (1.0, [3.0, 4.0], scst.expon.pdf(np.array([3.0, 4.0], dtype=np.float32), scale=1))
            ],
        }

    def test_distributions(self):
        """Test pdf."""
        for dist_name, cases in self.test_cases.items():  # For each distribution
            for case in cases:  # For each distribution's test cases
                with self.subTest(distribution=dist_name, params=case):
                    if dist_name == "Exponential":
                        distribution = Exponential(torch.tensor(case[0]))
                        k_tensor = torch.tensor(case[1], dtype=torch.float32)
                    else:
                        raise ValueError("Only exponential is supported in this unit test.")

                    pdf = PtdmPdfTensor().f(distribution, k_tensor)[0]

                    # Ensure return values are tensors
                    self.assertTrue(isinstance(pdf, torch.Tensor), "Probability output is not a tensor")

                    # Convert tensors to floats for comparison
                    if isinstance(case[1], float):
                        pdf = pdf.item()

                        # Check if computed values are approximately equal to expected values
                        self.assertAlmostEqual(pdf, case[-1], places=4, 
                                            msg=f"{dist_name}: Expected {case[-1]}, got {pdf}")

                    elif isinstance(case[1], list):
                        pdf = pdf.numpy()
                        self.assertTrue(
                           np.allclose(pdf, case[-1], atol=1e-5)
                        )

                    else:
                        raise ValueError("Unsupported data type.")
                    

if __name__ == "__main__":
    unittest.main()
