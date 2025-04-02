import os
import sys
import unittest
import torch
from itertools import product

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)

from hugging_face_wrapper.utils import drop_html_tags


class TestUtils(unittest.TestCase):
    def setUp(self):
        self.drop_html_tags_data = [
            ("Hello", "Hello"),
            ("Hello, <br>World", "Hello, World"),
            ("Hello, <a href='test.html'>test</a> World", "Hello, test World"),
        ]

    def test_drop_html_tags(self):
        for example, expected in self.drop_html_tags_data:
            actual = drop_html_tags(example)
            self.assertEqual(expected, actual, f"expected {expected} and actual {actual} do not match.")


if __name__ == "__main__":
    unittest.main()
