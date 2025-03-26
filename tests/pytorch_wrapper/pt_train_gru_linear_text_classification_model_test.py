"""
Dataset URL:https://huggingface.co/datasets/stanfordnlp/imdb
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

from pytorch_wrapper.ptn_gru_linear import PtnGRULinear
from pytorch_wrapper.pto_adamw import PtoAdamW
from pytorch_wrapper.ptv_hf_glove_dataset import PtvHfGloveDataset
from pytorch_wrapper.pt_data_loader import PtDataLoader
from pytorch_wrapper.pt_train_rnn_model import PtTrainRNNModel
from pytorch_wrapper.pt_save_model import PtSaveModel
from pytorch_wrapper.pt_load_model import PtLoadModel
from pytorch_wrapper.pto_lr_scheduler_reduce_on_plateau import PtoLrSchedulerReduceOnPlateau
from pytorch_wrapper.ptn_bce_with_logits_loss import PtnBCEWithLogitsLoss
from pytorch_wrapper.utils import set_seed


class TestPtTrainGRULinearTextClassificationModel(unittest.TestCase):

    @unittest.skipIf(os.getenv("RUN_SKIPPED_TESTS") != "1", "Skipping unless explicitly enabled")
    def setUp(self):
        set_seed(42)

        self.loss_fun =PtnBCEWithLogitsLoss().f(reduction="mean")[0]
        dataset_node = PtvHfGloveDataset()
        self.embedding_dim = 100
        self.max_seq_len = 300
        train_dataset = dataset_node.f(
            "imdb",  # dataset_name
            "train",  #split
            self.embedding_dim,
            self.max_seq_len,
            "text",  # sample field name in the dataset
            "label"  # label field name in the dataset
        )[0]
        test_dataset = dataset_node.f(
            "imdb",  # dataset_name
            "test",  #split
            self.embedding_dim,
            self.max_seq_len,
            "text",  # sample field name in the dataset
            "label"  # label field name in the dataset
        )[0]
        train_data_loader_node = PtDataLoader()
        test_data_loader_node = PtDataLoader()

        self.train_loader = train_data_loader_node.f(
            dataset=train_dataset,
            batch_size=128,
            shuffle=True,
            parameters='{"num_workers":1}'
        )[0]

        self.test_loader = test_data_loader_node.f(
            dataset=test_dataset,
            batch_size=128,
            shuffle=False,
            parameters='{"num_workers":1}'
        )[0]

        self.model_path = "gru_text_classification_test.pt"

        self.model_node = PtnGRULinear()
        self.model = self.model_node.f(
            self.embedding_dim, # input_size
            512, # hidden_size
            2, # num_layers
            True, # bias,
            True, # batch_first [B, Seq, Token]
            0.4, # dropout
            True, # bidirectional,
            1, # linear output size
            True, # linear bias
        )[0]

        self.optimizer = PtoAdamW().f(self.model,
                                      0.001,
                                      0.9, 0.999,
                                      weight_decay=0.01,
                                      amsgrad=False)[0]
        
        self.scheduler = PtoLrSchedulerReduceOnPlateau().f(self.optimizer,
                                                grace_period=10,
                                                gamma=0.5
                                               )[0]

        self.trainer = PtTrainRNNModel()
        self.save_model = PtSaveModel()
        self.load_model = PtLoadModel()
        self.linear_head = True
        self.epochs = 50

    @unittest.skipIf(os.getenv("RUN_SKIPPED_TESTS") != "1", "Skipping unless explicitly enabled")
    def test_1(self):

        trained_model = self.trainer.f(
                        self.model,
                        self.linear_head,
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
                        use_valid_token_mean=True,
                        val_loader=self.test_loader)[0]
        self.save_model.f(trained_model, self.model_path)


if __name__ == "__main__":
    unittest.main()
