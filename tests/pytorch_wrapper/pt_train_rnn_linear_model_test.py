import os
import sys
import unittest
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)

from pytorch_wrapper.ptn_rnn_linear import PtnRNNLinear
from pytorch_wrapper.pto_adam import PtoAdam
from pytorch_wrapper.ptv_sequential_tensor_dataset import PtvSequentialTensorDataset
from pytorch_wrapper.pt_data_loader import PtDataLoader
from pytorch_wrapper.pt_train_rnn_model import PtTrainRNNModel
from pytorch_wrapper.pt_save_model import PtSaveModel
from pytorch_wrapper.pt_load_model import PtLoadModel
from pytorch_wrapper.pto_lr_scheduler_step import PtoLrSchedulerStep
from pytorch_wrapper.ptn_mse_loss import PtnMSELoss
from pytorch_wrapper.utils import set_seed


class TestPtTrainRNNLinearModel(unittest.TestCase):
    def setUp(self):
        set_seed(49)

        self.loss_fun = PtnMSELoss().f(reduction="mean")[0]
        dataset_node = PtvSequentialTensorDataset()
        dataset = dataset_node.f(
                    torch.arange(0.01, 1.0, 0.01, dtype=torch.float32),
                    seq_len=4)[0]
        data_loader_node = PtDataLoader()

        self.train_loader = data_loader_node.f(
            dataset=dataset,
            batch_size=128,
            shuffle=True,
            parameters='{"num_workers":1}'
        )[0]

        self.model_node = PtnRNNLinear()
        self.model = self.model_node.f(
            1, # input_size
            32, # hidden_size
            4, # num_layers
            "tanh", # nonlinearity
            True, # bias,
            True, # batch_first [B, Seq, Token]
            0.0, # dropout
            False, # bidirectional,
            1, # linear output size
            True, # linear bias
        )[0]

        self.optimizer = PtoAdam().f(self.model, 0.001, 0.9, 0.999)[0]
        
        self.scheduler = PtoLrSchedulerStep().f(self.optimizer,
                                                100,  # steps
                                                0.1  # gamma
                                               )[0]

        self.trainer = PtTrainRNNModel()
        self.save_model = PtSaveModel()
        self.load_model = PtLoadModel()
        self.linear_head = True
        self.epochs = 200

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
                        classification_metrics=False,
                        use_valid_token_mean=False,
                        output_best_val_model=False)[0]
        self.save_model.f(trained_model, f"rnn_{self.epochs}_epochs.pt")

    def test_2(self):
        """
        Tests prediction.  Note that prediction can be off even if loss is close to 0.
        """
        loaded_model = self.load_model.f(self.model, f"rnn_{self.epochs}_epochs.pt" )[0]
        x = torch.tensor([[0.01, 0.02, 0.03, 0.04]], dtype=torch.float32).unsqueeze(-1)
        y_hat = loaded_model(x)
        # print(y_hat)


if __name__ == "__main__":
    unittest.main()
