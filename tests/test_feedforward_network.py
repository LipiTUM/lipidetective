"""Tests for FeedForwardNetwork class."""

import torch

from lipidetective.models.feedforward_network import FeedForwardNetwork


class TestFeedForwardNetworkInitialization:
    """Tests for FeedForwardNetwork initialization."""

    def test_initialization_with_peaks_input(self, regression_config):
        """Model should initialize with peaks input type."""
        model = FeedForwardNetwork(regression_config)

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

        model = FeedForwardNetwork(config)

        assert model.input_size == 1000

    def test_layers_created_from_config(self, regression_config):
        """Linear layers should match config sizes."""
        model = FeedForwardNetwork(regression_config)

        assert model.fc1.out_features == regression_config["feedforward"]["layer_1_size"]
        assert model.fc3.out_features == regression_config["feedforward"]["layer_3_size"]
        assert model.fc4.out_features == 3  # Always 3 outputs


class TestFeedForwardNetworkForward:
    """Tests for FeedForwardNetwork forward pass."""

    def test_forward_returns_correct_shape(self, regression_config):
        """Forward pass should return (batch_size, 3) tensor."""
        model = FeedForwardNetwork(regression_config)
        model.eval()

        batch_size = 4
        n_peaks = regression_config["input_embedding"]["n_peaks"]

        # Input shape: (batch, 2, n_peaks+1) - 2 channels for m/z and intensity
        x = torch.randn(batch_size, 2, n_peaks + 1)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (batch_size, 3)

    def test_forward_different_batch_sizes(self, regression_config):
        """Forward should work with different batch sizes."""
        model = FeedForwardNetwork(regression_config)
        model.eval()

        n_peaks = regression_config["input_embedding"]["n_peaks"]

        for batch_size in [1, 8, 32]:
            x = torch.randn(batch_size, 2, n_peaks + 1)
            with torch.no_grad():
                output = model(x)
            assert output.shape == (batch_size, 3)

    def test_forward_output_is_float(self, regression_config):
        """Output should be float tensor for regression."""
        model = FeedForwardNetwork(regression_config)
        model.eval()

        x = torch.randn(2, 2, regression_config["input_embedding"]["n_peaks"] + 1)

        with torch.no_grad():
            output = model(x)

        assert output.dtype == torch.float32

    def test_forward_no_nan_output(self, regression_config):
        """Forward pass should not produce NaN values."""
        model = FeedForwardNetwork(regression_config)
        model.eval()

        x = torch.randn(4, 2, regression_config["input_embedding"]["n_peaks"] + 1)

        with torch.no_grad():
            output = model(x)

        assert not torch.isnan(output).any()
