import os
import sys
import torch
import unittest

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)
from pytorch_wrapper.ptv_sequential_tensor_dataset import PtvSequentialTensorDataset
from pytorch_wrapper.pt_data_loader import PtDataLoader


class TestPtvSequentialTensorDataset(unittest.TestCase):
    def setUp(self):
        """Set up test data."""
        self.node = PtvSequentialTensorDataset()
        self.data = torch.arange(1, 100, dtype=torch.float32)

    def test_1(self):
        """Tests first seq """
        d = self.node.f(self.data, seq_len=5)[0]
        actual = next(iter(d))[0]
        expected = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32).unsqueeze(-1)
        self.assertTrue(
            torch.allclose(actual, expected, atol=4),
            f"Mismatch: actual: {actual}, expected: {expected}")
            
    def test_2(self):
        """Tests last seq """
        d = self.node.f(self.data, seq_len=5)[0]
        it = iter(d)
        for _ in range(94):
            next(it)

        actual = next(it)[0]
            
        # Get the tensor for the last seq
        expected = torch.tensor([95, 96, 97, 98, 99], dtype=torch.float32).unsqueeze(-1)
        self.assertEqual(actual.size(), expected.size())
        self.assertTrue(
            torch.allclose(actual, expected, atol=4),
            f"Mismatch: actual: {actual}, expected: {expected}")

    def test_3(self):
        """Tests last seq """
        self.data = torch.arange(0.01, 1, 0.01, dtype=torch.float32)
        d = self.node.f(self.data, seq_len=5)[0]
        it = iter(d)
        for _ in range(94):
            next(it)

        actual = next(it)[0]
            
        # Get the tensor for the last seq
        expected = torch.tensor([0.95, 0.96, 0.97, 0.98, 0.99], dtype=torch.float32).unsqueeze(-1)
        self.assertEqual(actual.size(), expected.size())
        self.assertTrue(
            torch.allclose(actual, expected, atol=4),
            f"Mismatch: actual: {actual}, expected: {expected}")

    def test_4_loader(self):
        """Tests loader """
        bs = 32
        seq_len = 5
        self.data = torch.arange(0.01, 1, 0.01, dtype=torch.float32)
        dataset = self.node.f(self.data, seq_len=seq_len)[0]
        data_loader_node = PtDataLoader()

        data_loader = data_loader_node.f(
            dataset=dataset,
            batch_size=bs,
            shuffle=True,
            parameters='{"num_workers":1}'
        )[0]

        it = iter(data_loader)
        x_batch, y_batch = next(it)
        self.assertTrue(x_batch.size() == (bs, seq_len, 1))
        self.assertTrue(y_batch.size() == (bs, seq_len, 1))


if __name__ == "__main__":
    unittest.main()
