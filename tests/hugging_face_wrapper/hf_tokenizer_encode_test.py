import os
import sys
import unittest
import torch
from itertools import product

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)

from hugging_face_wrapper.hf_tokenizer_encode import HfTokenizerEncode


class TestHfTokenizerEncode(unittest.TestCase):
    def setUp(self):
        self.node = HfTokenizerEncode()
        self.test_sentence_1d = "Hello, world"
        self.test_sentence_2d = [
            "a b c d e f g h i j k l m n o p q r s t u v w x y z",
            "a b c d e f g h i j k l"]
        self.models = {
            "bert-base-uncased": "BertTokenizer",
            "roberta-base": "RobertaTokenizer",
            "distilbert-base-uncased": "DistilBertTokenizer",
            "albert-base-v2": "AlbertTokenizer",
            "google/electra-small-discriminator": "ElectraTokenizer",
        }

    def test_tokenizer_variants(self):
        for model_name, tokenizer_name in self.models.items():
            for is_1d, padding, truncation, pad_method in product(
                [True, False], [True, False], [True, False], ["longest", "max_length"]
            ):
                
                if not is_1d and not padding:
                    continue  # Cannot batch ragged output into tensors

                with self.subTest(
                    tokenizer_name=tokenizer_name,
                    is_1d=is_1d,
                    padding=padding,
                    truncation=truncation,
                    pad_method=pad_method
                ):
                    sentence = self.test_sentence_1d if is_1d else self.test_sentence_2d
                    
                    if truncation:
                        max_length = 10
                    else:
                        max_length = 256

                    encoder = self.node.f(
                        model_name=model_name,
                        padding=padding,
                        padding_method=pad_method,
                        truncation=truncation,
                        max_length=max_length
                    )[0]
                    output = encoder(sentence=sentence)

                    input_ids = output["input_ids"]
                    mask = output["attention_mask"]

                    # shape agreement
                    self.assertEqual(input_ids.shape, mask.shape)

                    # correct batch size
                    self.assertEqual(input_ids.shape[0], 1 if is_1d else len(sentence))

                    # correct max length behavior
                    if not is_1d:
                        if truncation:
                            self.assertEqual(input_ids.shape[1], 10)
                        elif padding and pad_method == "longest":
                            self.assertEqual(input_ids[0].size(0), input_ids[1].size(0))
                        elif padding and pad_method == "max_length":
                            self.assertGreaterEqual(input_ids.shape[1], 256)


    def test_padding_false_1d(self):
        for model_name in self.models:
            with self.subTest(model=model_name):
                encoder = self.node.f(
                    model_name=model_name,
                    padding=False,
                    padding_method="longest",
                    truncation=False,
                    max_length=256
                )[0]
                output = encoder(sentence=self.test_sentence_1d)
                input_ids = output["input_ids"]
                self.assertEqual(input_ids.shape[0], 1)
                self.assertGreater(torch.count_nonzero(input_ids), 0)


if __name__ == "__main__":
    unittest.main()
