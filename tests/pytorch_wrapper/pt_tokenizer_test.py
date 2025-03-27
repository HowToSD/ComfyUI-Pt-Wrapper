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
from pytorch_wrapper.pt_tokenizer import PtTokenizer

class TestPtTokenizer(unittest.TestCase):
    def setUp(self):
        self.load_node = SpLoadModel()
        self.encode_node = SpEncode()
        self.tokenizer_node = PtTokenizer()

        self.sp = self.load_node.f("spiece.model")[0]

    def test_1(self):
        sentence = "Hello, world"
        encode = self.encode_node.f(
            spmodel=self.sp,
            padding=False,
            padding_method="longest",
            padding_value=0,
            truncation=False,
            max_length=256
        )[0]

        tokens, mask = self.tokenizer_node.f(encode=encode, text_list=sentence)

        expected_tokens = torch.tensor([8774, 6, 296], dtype=torch.long)
        expected_mask = torch.tensor([1, 1, 1], dtype=torch.long)
        self.assertTrue(torch.equal(tokens, expected_tokens))
        self.assertTrue(torch.equal(mask, expected_mask))


if __name__ == "__main__":
    unittest.main()
