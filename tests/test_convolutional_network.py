"""Tests for ConvolutionalNetwork class."""

import torch
from lipidetective.models.convolutional_network import ConvolutionalNetwork


class TestConvolutionalNetworkInitialization:
    """Tests for ConvolutionalNetwork initialization."""

    def test_initialization_with_peaks_input(self, regression_config):
        """Model should initialize with peaks input type."""
        model = ConvolutionalNetwork(regression_config)

        expected_input_size = regression_config["input_embedding"]["n_peaks"] + 1
        assert model.input_size == expected_input_size

    def test_initialization_with_spectrum_input(self, regression_config):
        """Model should initialize with spectrum (binned) input type."""
        config = regression_config.copy()
        config["input_embedding"] = {
            "type": "spectrum",
            "min_mz": 0,
            "max_mz": 1000,
            "precision": 1.0,
        }

        model = ConvolutionalNetwork(config)

        # Input size should be length of mz range
        assert model.input_size == 1000

    def test_layers_created_from_config(self, regression_config):
        """Convolutional and linear layers should match config."""
        model = ConvolutionalNetwork(regression_config)

        assert model.conv1.out_channels == regression_config["convolutional"]["channels_conv_1"]
        assert model.conv2.out_channels == regression_config["convolutional"]["channels_conv_2"]
        assert model.conv3.out_channels == regression_config["convolutional"]["channels_conv_3"]
        assert model.fc1.out_features == regression_config["convolutional"]["lin_layer_1"]
        assert model.fc2.out_features == regression_config["convolutional"]["lin_layer_2"]
        assert model.fc3.out_features == 3  # Always 3 outputs (headgroup + 2 fatty acids)


class TestConvolutionalNetworkForward:
    """Tests for ConvolutionalNetwork forward pass."""

    def test_forward_returns_correct_shape(self, regression_config):
        """Forward pass should return (batch_size, 3) tensor."""
        model = ConvolutionalNetwork(regression_config)
        model.eval()

        batch_size = 4
        n_peaks = regression_config["input_embedding"]["n_peaks"]

        # Input shape: (batch, 2, n_peaks+1)
        x = torch.randn(batch_size, 2, n_peaks + 1)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (batch_size, 3)

    def test_forward_different_batch_sizes(self, regression_config):
        """Forward should work with different batch sizes."""
        model = ConvolutionalNetwork(regression_config)
        model.eval()

        n_peaks = regression_config["input_embedding"]["n_peaks"]

        for batch_size in [1, 8, 16]:
            x = torch.randn(batch_size, 2, n_peaks + 1)
            with torch.no_grad():
                output = model(x)
            assert output.shape == (batch_size, 3)

    def test_forward_output_is_float(self, regression_config):
        """Output should be float tensor for regression."""
        model = ConvolutionalNetwork(regression_config)
        model.eval()

        x = torch.randn(2, 2, regression_config["input_embedding"]["n_peaks"] + 1)

        with torch.no_grad():
            output = model(x)

        assert output.dtype == torch.float32


class TestCalculateFc1Size:
    """Tests for FC1 size calculation."""

    def test_fc1_size_calculation(self, regression_config):
        """FC1 size should be correctly calculated from conv layer output."""
        model = ConvolutionalNetwork(regression_config)

        input_size = regression_config["input_embedding"]["n_peaks"] + 1
        fc1_size = model.calculate_fc1_size(input_size)

        # Should be a positive integer
        assert isinstance(fc1_size, int)
        assert fc1_size > 0

    def test_fc1_matches_layer_input(self, regression_config):
        """Calculated FC1 size should match the actual layer configuration."""
        model = ConvolutionalNetwork(regression_config)

        # The fc1 layer's in_features should use channels_3 (conv3 output) * spatial size
        expected_in_features = regression_config["convolutional"][
            "channels_conv_3"
        ] * model.calculate_fc1_size(model.input_size)
        assert model.fc1.in_features == expected_in_features
