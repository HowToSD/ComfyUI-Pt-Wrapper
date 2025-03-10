"""
Based on my earlier work:
https://github.com/hideyukiinada/cifar10/blob/master/project/keras_25
"""
from typing import List, Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleConvBlock(nn.Module):
    """
    pragma: skip_doc
    """

    def __init__(self, in_channels, out_channels, kernel_size, strides):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=strides,
            padding=padding,
            bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        torch.nn.init.kaiming_normal_(self.conv.weight,
                                      nonlinearity='relu')


    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class SingleConvBlockNoActivation(nn.Module):
    """
    pragma: skip_doc
    """

    def __init__(self, in_channels, out_channels, kernel_size, strides):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=strides,
            padding=padding,
            bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

        torch.nn.init.xavier_normal_(self.conv.weight)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class SingleResnetBlock(nn.Module):
    """
    pragma: skip_doc
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, downsample=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.downsample = downsample

        if self.downsample:
            self.single_conv_k1_s2_na = \
                SingleConvBlockNoActivation(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=1,
                    strides=2)  # Downsample

            self.first_conv_block = \
                SingleConvBlock(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=self.kernel_size,
                    strides=2)
        else:
            self.first_conv_block = \
                SingleConvBlock(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=self.kernel_size,
                    strides=1)
            
            if self.in_channels != self.out_channels:
                self.single_conv_k1_s1_na = \
                    SingleConvBlockNoActivation(
                        in_channels=self.in_channels,
                        out_channels=self.out_channels,
                        kernel_size=1,
                        strides=1)  # Channel change without downsampling

        self.second_conv = \
            SingleConvBlockNoActivation(
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                strides=1)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        original_tens = inputs
        tens = inputs

        if self.downsample:
            original_tens = self.single_conv_k1_s2_na(original_tens)
        elif self.in_channels != self.out_channels:
           original_tens = self.single_conv_k1_s1_na(original_tens)

        tens = self.first_conv_block(tens)

        # Second conv
        tens = self.second_conv(tens)
        tens = tens + original_tens
        tens = self.relu(tens)
        return tens


class NResnetBlocks(nn.Module):
    """
    pragma: skip_doc
    """

    def __init__(
            self,
            num_blocks,
            in_channels,
            out_channels,
            kernel_size=3,
            downsample=False):
        super().__init__()
        self.num_blocks = num_blocks
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.downsample = downsample

        if self.downsample:
            self.single_resnet_block = SingleResnetBlock(in_channels=self.in_channels,
                                                    out_channels=self.out_channels,
                                                    kernel_size=kernel_size,
                                                    downsample=True)
            block_count = self.num_blocks - 1
        else:
            block_count = self.num_blocks

        resnet_blocks = []
        for i in range(block_count):
            if i == 0 and not self.downsample:
                channels = self.in_channels
            else:
                channels = self.out_channels
            resnet_blocks.append(
                SingleResnetBlock(
                    in_channels=channels,
                    out_channels=self.out_channels,
                    kernel_size=self.kernel_size)
            )
        self.resnet_blocks = nn.ModuleList(resnet_blocks)

    def forward(self, inputs):
        tens = inputs
        if self.downsample:
            tens = self.single_resnet_block(tens)
        for block in self.resnet_blocks:
           tens = block(tens)

        return tens

class ResnetModel(nn.Module):
    """
    pragma: skip_doc
    """

    def __init__(self,
                 num_blocks,
                 in_channels,
                 input_height,
                 input_width,
                 output_dim):
        super().__init__()
        self.num_blocks = num_blocks
        self.in_channels = in_channels

        self.single_conv_block = SingleConvBlock(
            in_channels, out_channels=64, kernel_size=3, strides=1)

        self.n_resnet_blocks64 = NResnetBlocks(
            num_blocks, in_channels=64, out_channels=64, kernel_size=3)
        self.n_resnet_blocks128 = NResnetBlocks(
            num_blocks, in_channels=64, out_channels=128, kernel_size=3, downsample=True)
        self.n_resnet_blocks256 = NResnetBlocks(
            num_blocks, in_channels=128, out_channels=256, kernel_size=3, downsample=True)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling

        # Compute output size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, input_height, input_width)
            dummy_output = self._forward_features(dummy_input)
            feature_dim = dummy_output.shape[1] * dummy_output.shape[2] * dummy_output.shape[3]

        self.linear = nn.Linear(feature_dim, output_dim)  # Use computed feature size

    def _forward_features(self, inputs):
        """Forward pass to compute feature map dimensions."""
        x = self.single_conv_block(inputs)
        x = self.n_resnet_blocks64(x)
        x = self.n_resnet_blocks128(x)
        x = self.n_resnet_blocks256(x)
        return self.gap(x)  # Before flattening

    def forward(self, inputs):
        tens = self._forward_features(inputs)
        tens = torch.flatten(tens, start_dim=1)  # Flatten to (batch_size, computed_feature_dim)
        tens = self.linear(tens)
        return tens

