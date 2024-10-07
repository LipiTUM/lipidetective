import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FeedForwardNetwork(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        # +1 to peak input size for additional features of precursor mass and polarity
        if config['input_embedding']['type'] == 'peaks':
            self.input_size = (config['input_embedding']['n_peaks'] + 1)
        else:
            min_mz = config['input_embedding']['min_mz']
            max_mz = config['input_embedding']['max_mz']
            precision = config['input_embedding']['precision']

            self.input_size = len(np.arange(min_mz, max_mz, precision))

        self.layer_1 = config['feedforward']['layer_1_size']
        self.layer_2 = config['feedforward']['layer_2_size']
        self.layer_3 = config['feedforward']['layer_3_size']

        self.fc1 = nn.Linear(self.input_size, self.layer_1)
        self.fc2 = nn.Linear(2*self.layer_1, self.layer_2)
        self.fc3 = nn.Linear(self.layer_2, self.layer_3)
        self.fc4 = nn.Linear(self.layer_3, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))

        # Flatten separate vectors for intensity & m/z
        x = torch.flatten(x, 1)

        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x

