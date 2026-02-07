"""Unit tests for lightning_module.py."""

import os
import tempfile

import pytest
import torch

from lipidetective.helpers.logging import Evaluator
from lipidetective.models.convolutional_network import ConvolutionalNetwork
from lipidetective.models.feedforward_network import FeedForwardNetwork
from lipidetective.models.transformer_network import TransformerNetwork
from lipidetective.workflow.lightning_module import LightningModule


@pytest.fixture
def evaluator(lipid_library):
    """Create an Evaluator instance using shared lipid_library fixture."""
    return Evaluator(lipid_library)


@pytest.fixture
def sample_lipids():
    """Sample lipid names for testing."""
    return ["PC 16:0_18:1", "PE 18:0_18:2", "SM d18:1/16:0"]


class TestGetNeuralNetwork:
    """Tests for get_neural_network factory method."""

    def test_creates_transformer_network(self, lightning_config, evaluator):
        """Should create TransformerNetwork for 'transformer' model type."""
        lightning_config["model"] = "transformer"
        module = LightningModule(lightning_config, evaluator)

        assert isinstance(module.model, TransformerNetwork)

    def test_creates_convolutional_network(self, evaluator, regression_config):
        """Should create ConvolutionalNetwork for 'convolutional' model type."""
        config = {
            **regression_config,
            "model": "convolutional",
            "workflow": {"load_model": False, "tune": False},
            "training": {"batch": 4, "epochs": 2, "learning_rate": 0.001, "lr_step": 5},
        }
        module = LightningModule(config, evaluator)

        assert isinstance(module.model, ConvolutionalNetwork)

    def test_creates_feedforward_network(self, evaluator, regression_config):
        """Should create FeedForwardNetwork for 'feedforward' model type."""
        config = {
            **regression_config,
            "model": "feedforward",
            "workflow": {"load_model": False, "tune": False},
            "training": {"batch": 4, "epochs": 2, "learning_rate": 0.001, "lr_step": 5},
        }
        module = LightningModule(config, evaluator)

        assert isinstance(module.model, FeedForwardNetwork)

    def test_raises_for_unknown_model(self, lightning_config, evaluator):
        """Should raise ValueError for unknown model type."""
        lightning_config["model"] = "unknown_model"

        with pytest.raises(ValueError, match="Unknown model type"):
            LightningModule(lightning_config, evaluator)


class TestConfigureOptimizers:
    """Tests for configure_optimizers method."""

    def test_returns_optimizer_and_scheduler(self, lightning_config, evaluator):
        """Should return dict with optimizer and lr_scheduler."""
        module = LightningModule(lightning_config, evaluator)

        result = module.configure_optimizers()

        assert "optimizer" in result
        assert "lr_scheduler" in result
        assert isinstance(result["optimizer"], torch.optim.Adam)

    def test_uses_configured_learning_rate(self, lightning_config, evaluator):
        """Should use learning rate from config."""
        lightning_config["training"]["learning_rate"] = 0.01
        module = LightningModule(lightning_config, evaluator)

        result = module.configure_optimizers()
        optimizer = result["optimizer"]

        assert optimizer.defaults["lr"] == 0.01


class TestGetCustomLogger:
    """Tests for _get_custom_logger method."""

    def test_returns_none_when_no_logger(self, lightning_config, evaluator):
        """Should return None when module has no logger."""
        module = LightningModule(lightning_config, evaluator)

        result = module._get_custom_logger()

        assert result is None

    def test_returns_none_for_non_custom_logger(self, lightning_config, evaluator):
        """Should return None when logger name is not 'custom_logger'."""
        from unittest.mock import MagicMock

        module = LightningModule(lightning_config, evaluator)
        mock_logger = MagicMock()
        mock_logger.name = "tensorboard"
        module._logger = mock_logger

        result = module._get_custom_logger()

        assert result is None


class TestGetPredsVsLabels:
    """Tests for get_preds_vs_labels method."""

    def test_concatenates_correctly(self, lightning_config, evaluator):
        """Should concatenate epoch, batch, output, labels, and dataset path."""
        module = LightningModule(lightning_config, evaluator)
        module._current_epoch = 5

        output = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
        labels = torch.tensor([[7, 8, 9], [10, 11, 12]], dtype=torch.float32)
        dataset_path = torch.tensor([[0], [1]], dtype=torch.float32)

        result = module.get_preds_vs_labels(2, output, labels, dataset_path)

        assert result.shape == (
            2,
            9,
        )  # 2 samples, 2 (epoch+batch) + 3 (output) + 3 (labels) + 1 (path)


class TestGetTestPredsVsLabels:
    """Tests for get_test_preds_vs_labels method."""

    def test_concatenates_correctly(self, lightning_config, evaluator):
        """Should concatenate output, labels, dataset path, and confidence."""
        module = LightningModule(lightning_config, evaluator)

        output = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
        labels = torch.tensor([[7, 8, 9], [10, 11, 12]], dtype=torch.float32)
        dataset_path = torch.tensor([[0], [1]], dtype=torch.float32)
        confidence = torch.tensor([[0.9], [0.8]], dtype=torch.float32)

        result = module.get_test_preds_vs_labels(output, labels, dataset_path, confidence)

        assert result.shape == (2, 8)  # 2 samples, 3 + 3 + 1 + 1


class TestSaveModel:
    """Tests for save_model method."""

    def test_saves_model_to_file(self, lightning_config, evaluator):
        """Should save model state dict to output folder."""
        module = LightningModule(lightning_config, evaluator)

        with tempfile.TemporaryDirectory() as tmp_dir:
            module.save_model(tmp_dir)

            expected_file = os.path.join(tmp_dir, "lipidetective_model.pth")
            assert os.path.exists(expected_file)

            # Verify it's a valid state dict
            loaded = torch.load(expected_file, weights_only=True)
            assert isinstance(loaded, dict)


class TestLightningModuleInitialization:
    """Tests for LightningModule initialization."""

    def test_stores_config(self, lightning_config, evaluator):
        """Should store config."""
        module = LightningModule(lightning_config, evaluator)

        assert module.config == lightning_config

    def test_initializes_with_trainset_lipids(self, lightning_config, evaluator, sample_lipids):
        """Should initialize train metrics when trainset_lipids provided."""
        module = LightningModule(lightning_config, evaluator, trainset_lipids=sample_lipids)

        assert hasattr(module, "train_custom_accuracy")
        assert hasattr(module, "train_predictions")
        assert module.trainset_names == sample_lipids

    def test_initializes_with_valset_lipids(self, lightning_config, evaluator, sample_lipids):
        """Should initialize validation metrics when valset_lipids provided."""
        module = LightningModule(lightning_config, evaluator, valset_lipids=sample_lipids)

        assert hasattr(module, "val_custom_accuracy")
        assert hasattr(module, "val_predictions")
        assert module.valset_names == sample_lipids

    def test_initializes_with_testset_lipids(self, lightning_config, evaluator, sample_lipids):
        """Should initialize test metrics when testset_lipids provided."""
        module = LightningModule(lightning_config, evaluator, testset_lipids=sample_lipids)

        assert hasattr(module, "test_custom_accuracy")
        assert hasattr(module, "test_predictions")
        assert module.testset_names == sample_lipids

    def test_calculates_nr_epochs(self, lightning_config, evaluator):
        """Should calculate nr_epochs as epochs - 1."""
        lightning_config["training"]["epochs"] = 10
        module = LightningModule(lightning_config, evaluator)

        assert module.nr_epochs == 9


class TestRegressionModelMetrics:
    """Tests for regression model metric initialization."""

    def test_creates_mae_metrics_for_cnn(self, evaluator, regression_config, sample_lipids):
        """Should create MAE metrics for CNN model."""
        config = {
            **regression_config,
            "model": "convolutional",
            "workflow": {"load_model": False, "tune": False},
            "training": {"batch": 4, "epochs": 2, "learning_rate": 0.001, "lr_step": 5},
        }
        module = LightningModule(config, evaluator, trainset_lipids=sample_lipids)

        assert hasattr(module, "train_mae_hg")
        assert hasattr(module, "train_mae_fa1")
        assert hasattr(module, "train_mae_fa2")
        assert hasattr(module, "train_r2")

    def test_creates_val_metrics_when_valset_provided(
        self, evaluator, regression_config, sample_lipids
    ):
        """Should create validation MAE metrics when valset_lipids provided."""
        config = {
            **regression_config,
            "model": "feedforward",
            "workflow": {"load_model": False, "tune": False},
            "training": {"batch": 4, "epochs": 2, "learning_rate": 0.001, "lr_step": 5},
        }
        module = LightningModule(
            config, evaluator, trainset_lipids=sample_lipids, valset_lipids=sample_lipids
        )

        assert hasattr(module, "val_mae_hg")
        assert hasattr(module, "val_mae_fa1")
        assert hasattr(module, "val_mae_fa2")
        assert hasattr(module, "val_r2")
