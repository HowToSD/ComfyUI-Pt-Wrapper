"""
Dataset URL:https://huggingface.co/datasets/stanfordnlp/imdb

This unit test covers a rare test case where pretrained module is not frozen.
For a case where pretrained module is frozen, see:
tests/pytorch_wrapper/pt_train_fine_tune_classification_transformer_model_text_test.py

This script should produce validation accuracy around 0.9242 for the first epoch and around 0.9216 for the secod epoch.
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
from pytorch_wrapper.pt_load_model import PtLoadModel
from pytorch_wrapper.pto_lr_scheduler_cosine_annealing import PtoLrSchedulerCosineAnnealing
from pytorch_wrapper.ptn_bce_with_logits_loss import PtnBCEWithLogitsLoss
from pytorch_wrapper.utils import set_seed


class TestPtnHfFineTunedTextClassificationModel(unittest.TestCase):

    @unittest.skipIf(os.getenv("RUN_SKIPPED_TESTS") != "1", "Skipping unless explicitly enabled")
    def setUp(self):
        set_seed(42)

        self.loss_fun =PtnBCEWithLogitsLoss().f(reduction="mean")[0]

        self.max_length=512
        self.model_name = "distilbert-base-uncased"
        encode = HfTokenizerEncode().f(
            model_name=self.model_name,
            padding=True,
            padding_method="max_length",
            truncation=True,
            max_length=0  # Use model's max length
        )[0]

        train_dataset = HfDatasetWithTokenEncode(
            "imdb", # dataset_name
            "train", #split
            encode=encode,
            remove_html_tags=True,
            encode_return_dict=True
        )

        test_dataset = HfDatasetWithTokenEncode(
            "imdb", # dataset_name
            "test", #split
            encode=encode,
            remove_html_tags=True,
            encode_return_dict=True
        )

        train_data_loader_node = PtDataLoader()
        test_data_loader_node = PtDataLoader()

        self.train_loader = train_data_loader_node.f(
            dataset=train_dataset,
            batch_size=32,
            shuffle=True,
            parameters='{"num_workers":1}'
        )[0]

        self.test_loader = test_data_loader_node.f(
            dataset=test_dataset,
            batch_size=32,
            shuffle=False,
            parameters='{"num_workers":1}'
        )[0]

        self.model_path = "hf_fine_tuned_text_classification_model_test.pt"

        self.model = PtnHfFineTunedClassificationModel().f(
            self.model_name,
            use_mean_pooling=True,
            dropout=0.1
        )[0]

        self.optimizer = PtoAdamW().f(self.model, 1e-5, 0.9, 0.999,
                                      0.01,  # weight decay
                                      False  # amsgrad
                                      )[0]

        self.minimum_lr = 1e-9
        self.epochs = 2
        self.scheduler = PtoLrSchedulerCosineAnnealing().f(self.optimizer,
                                                           self.epochs,
                                                           self.minimum_lr)[0]

        self.trainer = PtTrainClassificationTransformerModel()
        self.save_model = PtSaveModel()
        self.load_model = PtLoadModel()
        self.linear_head = True


    @unittest.skipIf(os.getenv("RUN_SKIPPED_TESTS") != "1", "Skipping unless explicitly enabled")
    def test_1(self):

        trained_model = self.trainer.f(
                        self.model,
                        self.train_loader,
                        self.optimizer,
                        self.loss_fun,
                        self.epochs, # epochs
                        use_gpu=True,
                        early_stopping=False,
                        early_stopping_rounds=1,
                        scheduler=self.scheduler,
                        output_best_val_model=False,
                        classification_metrics=True,
                        val_loader=self.test_loader)[0]
        self.save_model.f(trained_model, self.model_path)


if __name__ == "__main__":
    unittest.main()
