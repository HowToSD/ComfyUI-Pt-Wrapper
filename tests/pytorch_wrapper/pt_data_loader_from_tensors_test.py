import os
import sys
import unittest
import torch

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)

from pytorch_wrapper.pt_data_loader_from_tensors import PtDataLoaderFromTensors

class TestPtDataLoader(unittest.TestCase):
    def setUp(self):
        """Set up test data."""
        self.data_loader_node = PtDataLoaderFromTensors()

    def test_with_1d(self):
        """Tests with transform"""
        x = torch.Tensor([0, 1, 2, 3])
        y = torch.Tensor([10, 20, 30, 40])

        bs = 2
        train_data = self.data_loader_node.f(
            x=x,
            y=y,
            batch_size=bs,
            shuffle=True
        )[0]

        x, y = next(iter(train_data))
        self.assertEqual(x.size(), torch.Size([bs]))
        self.assertEqual(y.size(), torch.Size([bs]))

    def test_with_2d(self):
        """Tests with transform"""
        x = torch.Tensor([0,1,2,3])
        x = torch.unsqueeze(x, -1)
        y = torch.Tensor([10,20,30,40])

        bs = 2
        train_data = self.data_loader_node.f(
            x=x,
            y=y,
            batch_size=bs,
            shuffle=True
        )[0]

        x, y = next(iter(train_data))
        self.assertEqual(x.size(), torch.Size([bs, 1]))
        self.assertEqual(y.size(), torch.Size([bs]))

    def test_with_2d_no_shuffle(self):
        """Tests with transform"""
        x = torch.Tensor([0,1,2,3])
        x = torch.unsqueeze(x, -1)
        y = torch.Tensor([10,20,30,40])

        bs = 2
        train_data = self.data_loader_node.f(
            x=x,
            y=y,
            batch_size=bs,
            shuffle=False
        )[0]

        x, y = next(iter(train_data))
        self.assertEqual(x[0], 0)
        self.assertEqual(y[0], 10)


if __name__ == "__main__":
    unittest.main()
