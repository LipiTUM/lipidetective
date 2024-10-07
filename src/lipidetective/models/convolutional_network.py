import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvolutionalNetwork(nn.Module):
    def __init__(self, config: dict):
        super(ConvolutionalNetwork, self).__init__()
        # +1 to peak input size for additional features of precursor mass and polarity
        if config['input_embedding']['type'] == 'peaks':
            self.input_size = (config['input_embedding']['n_peaks'] + 1)
        else:
            min_mz = config['input_embedding']['min_mz']
            max_mz = config['input_embedding']['max_mz']
            precision = config['input_embedding']['precision']

            self.input_size = len(np.arange(min_mz, max_mz, precision))

        self.config = config

        # Extract all relevant layer information from config file
        self.channels_1 = config['convolutional']['channels_conv_1']
        self.channels_2 = config['convolutional']['channels_conv_2']
        self.channels_3 = config['convolutional']['channels_conv_3']

        self.kernel_1 = config['convolutional']['kernel_size_1']
        self.kernel_2 = config['convolutional']['kernel_size_2']
        self.kernel_3 = config['convolutional']['kernel_size_3']

        self.stride_1 = config['convolutional']['stride_1']
        self.stride_2 = config['convolutional']['stride_2']
        self.stride_3 = config['convolutional']['stride_3']

        self.lin_1 = config['convolutional']['lin_layer_1']
        self.lin_2 = config['convolutional']['lin_layer_2']

        # Layers
        self.conv1 = nn.Conv2d(1, self.channels_1, kernel_size=tuple(self.kernel_1), stride=self.stride_1)
        self.conv2 = nn.Conv2d(self.channels_1, self.channels_2, kernel_size=tuple(self.kernel_2), stride=self.stride_2)
        self.conv3 = nn.Conv2d(self.channels_2, self.channels_3, kernel_size=tuple(self.kernel_3), stride=self.stride_3)

        fc_1_size = self.calculate_fc1_size(self.input_size)

        self.fc1 = nn.Linear(self.channels_2 * fc_1_size, self.lin_1)
        self.fc2 = nn.Linear(self.lin_1, self.lin_2)
        self.fc3 = nn.Linear(self.lin_2, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the convolutional network with three convolutional layers and pooling layers,
        followed by three fully connected linear layers.

        Args:
            x (torch.Tensor): input tensor of features with shape (batch_size, 2, n_peaks+1). Dimension 1 is size 2 as the tensor contains the m/z and intensity values of each peak. Dimension 2 is size n_peaks + 1 as the measurement mode (-1 for negative and +1 for positive) and the precursor mass are added.

        Returns:
            torch.Tensor: output of the convolutional network with shape (batch_size, 3). Corresponds to the three
            masses of the lipid components (headgroup and two side chains) that are supposed to be predicted.
        """

        x = x.unsqueeze(1)

        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=(1, 2))

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=(1, 2))

        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=(1, 2))

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def calculate_fc1_size(self, len_input_spectrum: int) -> int:
        """
        Calculates the size of the output after the convolutional layers so that the size of the first fully connected
        layer can be set accordingly.

        Args:
            len_input_spectrum (int): length of the input spectrum

        Returns:
            int: size of the in_features of the first fully connected layer

        """
        output_size_1 = ((len_input_spectrum - self.kernel_1[1]) / self.stride_1) + 1
        output_size_2 = ((math.floor(output_size_1)) / 2)  # pooling 1
        output_size_3 = ((math.floor(output_size_2) - self.kernel_2[1]) / self.stride_2) + 1
        output_size_4 = ((math.floor(output_size_3)) / 2)  # pooling 2
        output_size_5 = ((math.floor(output_size_4) - self.kernel_3[1]) / self.stride_3) + 1
        fc1_size = ((math.floor(output_size_5)) / 2)  # pooling 3

        return int(fc1_size)
