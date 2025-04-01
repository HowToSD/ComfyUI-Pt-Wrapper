import os
import sys
import unittest
import torch

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)
from pytorch_wrapper.ptn_hf_lora_classification_model_def import HfLoraClassificationModel
from hugging_face_wrapper.hf_tokenizer_encode import HfTokenizerEncode


class TestHfLoraClassificationModel(unittest.TestCase):
    def setUp(self):
        """Setup shared test conditions."""
        self.text = [
            "I really liked the scenary.",
            "I did not like the vision of the creator."
        ]
        self.model_names = [
            "distilbert-base-uncased",
            "bert-base-uncased",
            "roberta-base",
            "albert-base-v2",
            "google/electra-base-discriminator"
        ]

    def test_all_models(self):
        """Run output shape test across all supported models."""
        for model_name in self.model_names:
            with self.subTest(model_name=model_name):
                encode = HfTokenizerEncode().f(
                    model_name=model_name,
                    padding=True,
                    padding_method="max_length",
                    truncation=True,
                    max_length=0  # Use model's max length
                )[0]

                model = HfLoraClassificationModel(
                    model_name,
                    use_mean_pooling=True,
                    dropout=0.1
                )

                token_dict = encode(self.text)
                out = model((token_dict["input_ids"], token_dict["attention_mask"]))
                self.assertEqual(out.size(), torch.Size([2, 1]))


if __name__ == "__main__":
    unittest.main()
