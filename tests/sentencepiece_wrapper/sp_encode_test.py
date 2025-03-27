import os
import sys
import unittest
import sentencepiece as spm
import torch

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)

from sentencepiece_wrapper.sp_load_model import SpLoadModel
from sentencepiece_wrapper.sp_encode import SpEncode


class TestSpEncode(unittest.TestCase):
    def setUp(self):
        self.load_node = SpLoadModel()
        self.node = SpEncode()
        self.sp = self.load_node.f("spiece.model")[0]

    def test_1d_no_padding(self):
        sentence = "Hello, world"
        encoder = self.node.f(
            spmodel=self.sp,
            padding=False,
            padding_method="longest",
            padding_value=0,
            truncation=False,
            max_length=256
        )[0]
        tokens, mask = encoder(sentence=sentence)
        print(tokens)
        expected_tokens = torch.tensor([8774, 6, 296], dtype=torch.long)
        expected_mask = torch.tensor([1, 1, 1], dtype=torch.long)
        self.assertTrue(torch.equal(tokens, expected_tokens))
        self.assertTrue(torch.equal(mask, expected_mask))

    def test_2d_no_padding(self):
        sentence = ["Hello,", "world"]
        encoder = self.node.f(
            spmodel=self.sp,
            padding=False,
            padding_method="longest",
            padding_value=0,
            truncation=False,
            max_length=256
        )[0]
        tokens, masks = encoder(sentence=sentence)

        expected_tokens = [
            torch.tensor([8774, 6], dtype=torch.long),
            torch.tensor([296], dtype=torch.long)
        ]
        expected_masks = [
            torch.tensor([1, 1], dtype=torch.long),
            torch.tensor([1], dtype=torch.long)
        ]
        for t, e in zip(tokens, expected_tokens):
            self.assertTrue(torch.equal(t, e))
        for m, e in zip(masks, expected_masks):
            self.assertTrue(torch.equal(m, e))

    def test_1d_with_padding(self):
        sentence = "Hello, world"
        encoder = self.node.f(
            spmodel=self.sp,
            padding=True,
            padding_method="longest",
            padding_value=0,
            truncation=False,
            max_length=256
        )[0]
        tokens, mask = encoder(sentence=sentence)
        expected_tokens = torch.tensor([8774, 6, 296], dtype=torch.long)
        expected_mask = torch.tensor([1, 1, 1], dtype=torch.long)
        self.assertTrue(torch.equal(tokens, expected_tokens))
        self.assertTrue(torch.equal(mask, expected_mask))

    def test_2d_with_padding(self):
        sentence = ["Hello,", "world"]
        encoder = self.node.f(
            spmodel=self.sp,
            padding=True,
            padding_method="longest",
            padding_value=0,
            truncation=False,
            max_length=256
        )[0]
        tokens, mask = encoder(sentence=sentence)

        expected_tokens = torch.tensor([
            [8774, 6],
            [296, 0]
        ], dtype=torch.long)
        expected_mask = torch.tensor([
            [1, 1],
            [1, 0]
        ], dtype=torch.long)
        self.assertTrue(torch.equal(tokens, expected_tokens))
        self.assertTrue(torch.equal(mask, expected_mask))

    def test_1d_truncation(self):
        sentence = "Hello, world"
        encoder = self.node.f(
            spmodel=self.sp,
            padding=False,
            padding_method="longest",
            padding_value=0,
            truncation=True,
            max_length=2
        )[0]
        tokens, mask = encoder(sentence=sentence)

        expected_tokens = torch.tensor([8774, 6], dtype=torch.long)
        expected_mask = torch.tensor([1, 1], dtype=torch.long)
        self.assertTrue(torch.equal(tokens, expected_tokens))
        self.assertTrue(torch.equal(mask, expected_mask))

    def test_2d_truncation(self):
        sentence = ["Hello,", "world"]
        encoder = self.node.f(
            spmodel=self.sp,
            padding=False,
            padding_method="longest",
            padding_value=0,
            truncation=True,
            max_length=1
        )[0]
        tokens, masks = encoder(sentence=sentence)

        expected_tokens = [
            torch.tensor([8774], dtype=torch.long),
            torch.tensor([296], dtype=torch.long)
        ]
        expected_masks = [
            torch.tensor([1], dtype=torch.long),
            torch.tensor([1], dtype=torch.long)
        ]
        for t, e in zip(tokens, expected_tokens):
            self.assertTrue(torch.equal(t, e))
        for m, e in zip(masks, expected_masks):
            self.assertTrue(torch.equal(m, e))


if __name__ == "__main__":
    unittest.main()
