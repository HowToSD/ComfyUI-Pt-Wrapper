import os
import sys
import unittest
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)
from pytorch_wrapper.pt_predict_regression_model import PtPredictRegressionModel


class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1, True)
    def forward(self, inputs):
        return self.linear(inputs)

class TestPtPredictionRegressionModel(unittest.TestCase):

    def setUp(self):
        """Set up test instance."""
        self.model = TestModel()
        self.x = torch.arange(0, 11, dtype=torch.float32)

    def test_prediction(self):
        """
        Tests prediction.  

        Note that this test requires a trained model file on the file system.
        Unit test will fail if you run this for the first time.
        Run the unit test for training to generate the weight, and re-run this unit test.
        """
        self.predict = PtPredictRegressionModel().f(
            model = self.model,
            inputs = self.x,
            use_gpu = True
        )[0]

        self.assertTrue(self.predict.size() == torch.unsqueeze(self.x,-1).size())

if __name__ == "__main__":
    unittest.main()
