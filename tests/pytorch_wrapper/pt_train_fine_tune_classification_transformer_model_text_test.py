"""
Dataset URL:https://huggingface.co/datasets/stanfordnlp/imdb

This unit test covers a test case where pretrained module is not frozen.
For a case where pretrained module is frozen, see:
tests/pytorch_wrapper/pt_train_fine_tune_classification_transformer_model_text_test.py

For DistilBERT, this script should produce validation accuracy around 0.9242 for the first epoch and around 0.9216 for the secod epoch.
"""
import os
import sys
import unittest
import torch
import torch.nn as nn
import random
import numpy as np

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)

from pytorch_wrapper.ptn_hf_fine_tuned_classification_model import PtnHfFineTunedClassificationModel
from hugging_face_wrapper.hf_dataset_with_token_encode import HfDatasetWithTokenEncode
from hugging_face_wrapper.hf_tokenizer_encode import HfTokenizerEncode
from pytorch_wrapper.pto_adamw import PtoAdamW
from pytorch_wrapper.pt_data_loader import PtDataLoader
from pytorch_wrapper.pt_train_classification_transformer_model import PtTrainClassificationTransformerModel
from pytorch_wrapper.pt_save_model import PtSaveModel
from pytorch_wrapper.pto_lr_scheduler_cosine_annealing import PtoLrSchedulerCosineAnnealing
from pytorch_wrapper.ptn_bce_with_logits_loss import PtnBCEWithLogitsLoss
from pytorch_wrapper.utils import set_seed


class TestPtnHfFineTunedTextClassificationModel(unittest.TestCase):

    @unittest.skipIf(os.getenv("RUN_SKIPPED_TESTS") != "1", "Skipping unless explicitly enabled")
    def setUp(self):
        set_seed(42)

        self.loss_fun = PtnBCEWithLogitsLoss().f(reduction="mean")[0]
        self.batch_size = 32
        self.epochs = 2
        self.minimum_lr = 1e-9
        self.models = {
            "bert-base-uncased": "BertTokenizer",
            "distilbert-base-uncased": "DistilBertTokenizer",
            "albert-base-v2": "AlbertTokenizer",
            "roberta-base": "RobertaTokenizer",
            # "google/electra-small-discriminator": "ElectraTokenizer",
        }
        self.model_path = "hf_fine_tuned_text_classification_model_test.pt"
        self.linear_head = True

    @unittest.skipIf(os.getenv("RUN_SKIPPED_TESTS") != "1", "Skipping unless explicitly enabled")
    def test_all_models(self):
        for model_name in self.models:
            print(f"\n=== Testing model: {model_name} ===")

            encode = HfTokenizerEncode().f(
                model_name=model_name,
                padding=True,
                padding_method="max_length",
                truncation=True,
                max_length=0  # Use model's max length
            )[0]

            train_dataset = HfDatasetWithTokenEncode(
                "imdb", "train", encode=encode,
                remove_html_tags=True,
                encode_return_dict=True
            )

            test_dataset = HfDatasetWithTokenEncode(
                "imdb", "test", encode=encode,
                remove_html_tags=True,
                encode_return_dict=True
            )

            train_loader = PtDataLoader().f(
                dataset=train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                parameters='{"num_workers":1}'
            )[0]

            test_loader = PtDataLoader().f(
                dataset=test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                parameters='{"num_workers":1}'
            )[0]

            model = PtnHfFineTunedClassificationModel().f(
                model_name,
                use_mean_pooling=True,
                dropout=0.1
            )[0]

            optimizer = PtoAdamW().f(
                model, 1e-5, 0.9, 0.999,
                0.01,  # weight decay
                False  # amsgrad
            )[0]

            scheduler = PtoLrSchedulerCosineAnnealing().f(
                optimizer,
                self.epochs,
                self.minimum_lr
            )[0]

            trainer = PtTrainClassificationTransformerModel()

            trained_model = trainer.f(
                model,
                train_loader,
                optimizer,
                self.loss_fun,
                self.epochs,
                use_gpu=True,
                early_stopping=False,
                early_stopping_rounds=1,
                scheduler=scheduler,
                output_best_val_model=False,
                classification_metrics=True,
                val_loader=test_loader
            )[0]

            PtSaveModel().f(trained_model, self.model_path)

            # Clean up to free memory and VRAM
            del model, trained_model, optimizer, scheduler, trainer
            del train_loader, test_loader, train_dataset, test_dataset, encode
            torch.cuda.empty_cache()
            import gc
            gc.collect()


if __name__ == "__main__":
    unittest.main()
