import os
import sys
import unittest
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)

from pytorch_wrapper.ptn_rnn import PtnRNN
from pytorch_wrapper.pto_adam import PtoAdam
from pytorch_wrapper.ptv_sequential_tensor_dataset import PtvSequentialTensorDataset
from pytorch_wrapper.pt_data_loader import PtDataLoader
from pytorch_wrapper.pt_train_rnn_model import PtTrainRNNModel
from pytorch_wrapper.pt_save_model import PtSaveModel
from pytorch_wrapper.pt_load_model import PtLoadModel
from pytorch_wrapper.pto_lr_scheduler_step import PtoLrSchedulerStep
from pytorch_wrapper.ptn_mse_loss import PtnMSELoss
from pytorch_wrapper.utils import set_seed


class TestPtTrainRNNModel(unittest.TestCase):
    def setUp(self):
        set_seed(43)

        self.loss_fun = PtnMSELoss().f(reduction="mean")[0]
        dataset_node = PtvSequentialTensorDataset()
        dataset = dataset_node.f(
                    torch.arange(1, 10, dtype=torch.float32) / 100.0,
                    4)[0]
        data_loader_node = PtDataLoader()

        self.train_loader = data_loader_node.f(
            dataset=dataset,
            batch_size=32,
            shuffle=True,
            parameters='{"num_workers":1}'
        )[0]

        self.model_node = PtnRNN()
        self.model = self.model_node.f(
            1, # input_size
            1, # hidden_size
            1, # num_layers
            "tanh", # nonlinearity
            True, # bias,
            True, # batch_first [B, Seq, Token]
            0.0, # dropout
            False # bidirectional
        )[0]

        self.optimizer = PtoAdam().f(self.model, 0.1, 0.9, 0.999)[0]
        
        self.scheduler = PtoLrSchedulerStep().f(self.optimizer,
                                                100,  # steps
                                                0.1  # gamma
                                               )[0]

        self.trainer = PtTrainRNNModel()
        self.save_model = PtSaveModel()
        self.load_model = PtLoadModel()
        self.epochs = 50
        self.linear_head = False

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
                        classification_metrics=False,
                        use_valid_token_mean=True)[0]
        self.save_model.f(trained_model, f"rnn_{self.epochs}_epochs.pt")

    def test_2(self):
        """
        Tests prediction. Note that due to a hidden size set to 1, the model sometimes fails to
        learn the sequence depending on the initial random weight values, so you may
        see the output that is far off from 0.05 here.
        """
        loaded_model = self.load_model.f(self.model, f"rnn_{self.epochs}_epochs.pt" )[0]
        x = (torch.tensor([[1, 2, 3, 4]], dtype=torch.float32) / 100).unsqueeze(-1)
        y_hat = loaded_model(x)
        # print(y_hat)

if __name__ == "__main__":
    unittest.main()
