import os
import sys
import unittest
import sentencepiece as spm

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)
from sentencepiece_wrapper.sp_load_model import SpLoadModel


class TestSpLoadModel(unittest.TestCase):
    def setUp(self):
        """Set up test data."""
        self.node = SpLoadModel()


    def test_1(self):
        """Tests loading."""
        sp = self.node.f("spiece.model")[0]
        
        # Check tokenization
        s = "Hello, world"
        actual = sp.encode(s, out_type=int)  # Token IDs
        expected = [8774, 6, 296]
        self.assertEqual(expected, actual, f"expected {expected} and actual {actual} do not match.")

if __name__ == "__main__":
    unittest.main()
