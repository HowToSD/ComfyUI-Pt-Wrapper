import os
import sys
import unittest
import torch

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)

from pytorch_wrapper.ptd_student_t import PtdStudentT

class TestPtdStudentT(unittest.TestCase):
    def setUp(self):
        """Set up test instance."""
        self.node = PtdStudentT()

    def test_student_t_distribution_scalar(self):
        """Test instantiation of student_t distribution."""
        dist = self.node.f("10.0", "0.0","1.0")[0]
        self.assertTrue(isinstance(dist, torch.distributions.studentT.StudentT), "Probability distribution is not StudentT")

    def test_student_t_distribution_1d(self):
        """Test instantiation of student_t distribution."""
        dist = self.node.f("(10.0,)", "(0.0,)", "(1.0,)")[0]
        self.assertTrue(isinstance(dist, torch.distributions.studentT.StudentT), "Probability distribution is not StudentT")

    def test_student_t_distribution_2d(self):
        """Test instantiation of student_t distribution."""
        dist = self.node.f("(10.0, 11.0)", "(155.0,170.0)","(8.0,10.0)")[0]
        self.assertTrue(isinstance(dist, torch.distributions.studentT.StudentT), "Probability distribution is not StudentT")

   
if __name__ == "__main__":
    unittest.main()
