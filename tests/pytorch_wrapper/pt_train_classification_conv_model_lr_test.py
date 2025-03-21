import os
import sys
import unittest
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_ROOT = os.path.join(PROJECT_ROOT, "modules")
sys.path.append(MODULE_ROOT)

from pytorch_wrapper.ptn_conv_model import PtnConvModel
from pytorch_wrapper.pto_adam import PtoAdam
from pytorch_wrapper.ptv_dataset import PtvDataset
from pytorch_wrapper.ptv_transforms_to_tensor import PtvTransformsToTensor
from pytorch_wrapper.pt_data_loader import PtDataLoader
from pytorch_wrapper.pt_train_classification_model_lr import PtTrainClassificationModelLr
from pytorch_wrapper.pt_save_model import PtSaveModel
from pytorch_wrapper.pto_lr_scheduler_step import PtoLrSchedulerStep


class TestPtTrainClassificationConvModelLr(unittest.TestCase):
    def setUp(self):
        """Set up test instance."""
        compose_node = PtvTransformsToTensor()
        transform = compose_node.f()[0]
        dataset_node = PtvDataset()
        dataset = dataset_node.f("FashionMNIST",
                    True, # download
                    "", # root
                    '{"train": True}',
                    transform=transform)[0]
        data_loader_node = PtDataLoader()
        self.train_loader = data_loader_node.f(
            dataset=dataset,
            batch_size=32,
            shuffle=True,
            parameters='{"num_workers":1}'
        )[0]

        self.model_node = PtnConvModel()
        self.model = self.model_node.f(
                 input_dim="(1, 28, 28)",
                 penultimate_dim=0,
                 output_dim=10,
                 channel_list="[32,64,128,256]",
                 kernel_size_list="[3,3,3,3]",
                 padding_list="[1,1,1,1]",
                 downsample_list="[True,True,True,True]"
        )[0]

        self.optimizer = PtoAdam().f(self.model, 0.001, 0.9, 0.999)[0]
        
        self.scheduler = PtoLrSchedulerStep().f(self.optimizer,
                                                3,  # steps
                                                0.1  # gamma
                                               )[0]

        self.trainer = PtTrainClassificationModelLr()
        self.save_model = PtSaveModel()

    def test_1(self):
        epochs = 4
        trained_model = self.trainer.f(
                        self.model,
                        self.train_loader,
                        self.optimizer,
                        self.scheduler,
                        epochs, # epochs
                        use_gpu=True,
                        early_stopping=False,
                        early_stopping_rounds=1,
                        output_best_val_model=False)[0]
        self.save_model.f(trained_model, f"fashion_mnist_{epochs}_epochs_conv.pt")


if __name__ == "__main__":
    unittest.main()
