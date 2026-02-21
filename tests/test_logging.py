"""Tests for logging module - Evaluator and CustomAccuracy classes."""

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend before importing pyplot

import os

import numpy as np
import pytest
import torch

from lipidetective.helpers.logging import CustomAccuracy, CustomLogger, Evaluator


class TestEvaluatorInitialization:
    """Tests for Evaluator initialization."""

    def test_initialization_with_library(self, lipid_library):
        """Evaluator should initialize with a LipidLibrary."""
        evaluator = Evaluator(lipid_library)
        assert evaluator.library is lipid_library


class TestEvaluatorFindNearestHeadgroup:
    """Tests for find_nearest_headgroup method."""

    def test_finds_nearest_headgroup(self, lipid_library):
        """Should find headgroup with closest normalized mass."""
        evaluator = Evaluator(lipid_library)

        # Use actual normalized mass from library
        sample_hg = list(lipid_library.headgroups_mass_norm.keys())[0]
        sample_value = lipid_library.headgroups_mass_norm[sample_hg]

        found_hg, found_val = evaluator.find_nearest_headgroup(sample_value)

        assert found_hg == sample_hg
        assert found_val == sample_value

    def test_finds_approximate_match(self, lipid_library):
        """Should find closest match for approximate value."""
        evaluator = Evaluator(lipid_library)

        # Get a value slightly off from a known headgroup
        sample_hg = list(lipid_library.headgroups_mass_norm.keys())[0]
        sample_value = lipid_library.headgroups_mass_norm[sample_hg]
        approximate_value = sample_value + 0.001

        found_hg, _ = evaluator.find_nearest_headgroup(approximate_value)

        # Should still find the closest headgroup
        assert found_hg == sample_hg


class TestEvaluatorFindNearestSideChain:
    """Tests for find_nearest_side_chain method."""

    def test_finds_nearest_side_chain(self, lipid_library):
        """Should find side chain with closest normalized mass."""
        evaluator = Evaluator(lipid_library)

        sample_sc = list(lipid_library.side_chains_mass_norm.keys())[0]
        sample_value = lipid_library.side_chains_mass_norm[sample_sc]

        found_sc, found_val = evaluator.find_nearest_side_chain(sample_value)

        assert found_sc == sample_sc
        assert found_val == sample_value


class TestEvaluatorRegressionAccuracy:
    """Tests for evaluate_regression_accuracy method."""

    def test_evaluates_correct_predictions(self, lipid_library):
        """Should count fully correct predictions."""
        evaluator = Evaluator(lipid_library)

        # Get actual lipid components from library
        sample_lipid = list(lipid_library.molecular_lipid_species.keys())[0]
        lipid_data = lipid_library.molecular_lipid_species[sample_lipid]

        headgroup = lipid_data["headgroup"]
        fa1 = lipid_data["fatty_acid_sn1"]
        fa2 = lipid_data["fatty_acid_sn2"]

        # Create prediction that exactly matches the normalized values
        predictions = [
            torch.tensor(
                [
                    [lipid_library.headgroups_mass_norm[headgroup]],
                    [lipid_library.side_chains_mass_norm[fa1]],
                    [lipid_library.side_chains_mass_norm[fa2]],
                ]
            ).T
        ]

        lipid_info = {
            "headgroup": [headgroup],
            "fatty_acid_sn1": [fa1],
            "fatty_acid_sn2": [fa2],
        }

        nr_correct, total = evaluator.evaluate_regression_accuracy(predictions, lipid_info)

        assert nr_correct == 1
        assert total == 1


class TestCustomAccuracyInitialization:
    """Tests for CustomAccuracy metric initialization."""

    def test_initialization_creates_states(self, lipid_library):
        """CustomAccuracy should initialize with correct states."""
        evaluator = Evaluator(lipid_library)
        lipid_names = ["PC 16:0_18:1", "PE 18:0_18:2", "noise_spectrum"]

        metric = CustomAccuracy(evaluator, lipid_names)

        assert hasattr(metric, "correct")
        assert hasattr(metric, "accuracy_sum")
        assert hasattr(metric, "total")
        assert hasattr(metric, "confusion_matrix")

    def test_lipid_name_dict_created(self, lipid_library):
        """Should create lipid name to index mapping."""
        evaluator = Evaluator(lipid_library)
        lipid_names = ["PC 16:0_18:1", "PE 18:0_18:2"]

        metric = CustomAccuracy(evaluator, lipid_names)

        assert metric.lipid_name_dict["PC 16:0_18:1"] == 0
        assert metric.lipid_name_dict["PE 18:0_18:2"] == 1


class TestCustomAccuracyCompute:
    """Tests for CustomAccuracy compute method."""

    def test_compute_returns_accuracy_tensor(self, lipid_library):
        """Compute should return tensor with accuracy and mean accuracy."""
        import warnings

        evaluator = Evaluator(lipid_library)
        lipid_names = ["PC 16:0_18:1", "noise_spectrum"]

        metric = CustomAccuracy(evaluator, lipid_names)

        # Manually set some state values (bypassing update() for isolated testing)
        metric.correct = torch.tensor(5)
        metric.total = torch.tensor(10)
        metric.accuracy_sum = torch.tensor(7.5)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = metric.compute()

        assert result.shape == (2,)
        assert result[0].item() == pytest.approx(0.5)  # 5/10
        assert result[1].item() == pytest.approx(0.75)  # 7.5/10

    def test_confusion_matrix_shape(self, lipid_library):
        """Confusion matrix should have correct shape."""
        evaluator = Evaluator(lipid_library)
        lipid_names = ["PC 16:0_18:1", "PE 18:0_18:2", "noise_spectrum"]

        metric = CustomAccuracy(evaluator, lipid_names)

        # Matrix is (n_lipids + 1, n_lipids) for "Other" predictions
        expected_shape = (len(lipid_names) + 1, len(lipid_names))
        assert metric.confusion_matrix.shape == expected_shape

    def test_get_confusion_matrix(self, lipid_library):
        """Should return the confusion matrix."""
        evaluator = Evaluator(lipid_library)
        lipid_names = ["PC 16:0_18:1"]

        metric = CustomAccuracy(evaluator, lipid_names)

        matrix = metric.get_confusion_matrix()
        assert matrix is metric.confusion_matrix


class TestEvaluatorTransformerAccuracy:
    """Tests for evaluate_custom_transformer_accuracy method."""

    def test_evaluates_identical_predictions(self, lipid_library):
        """Should return 100% accuracy for identical predictions."""
        evaluator = Evaluator(lipid_library)

        # Create token sequences for "PC 16:0_18:1"
        token_tensor, _ = lipid_library.get_transformer_label(
            "PC 16:0_18:1", "[M+H]+", output_seq_length=15
        )

        predictions = [token_tensor[1:]]  # Skip SOS
        labels = [token_tensor[1:]]
        lipid_name_dict = {"PC 16:0_18:1": 0, "noise_spectrum": 1}

        accuracy_sum, nr_correct, total, matrix = evaluator.evaluate_custom_transformer_accuracy(
            predictions, labels, lipid_name_dict, is_last_epoch=False
        )

        assert nr_correct == 1
        assert total == 1
        assert accuracy_sum == 1.0

    def test_evaluates_different_predictions(self, lipid_library):
        """Should return lower accuracy for different predictions."""
        evaluator = Evaluator(lipid_library)

        pred_tensor, _ = lipid_library.get_transformer_label(
            "PE 16:0_18:1", "[M+H]+", output_seq_length=15
        )
        label_tensor, _ = lipid_library.get_transformer_label(
            "PC 16:0_18:1", "[M+H]+", output_seq_length=15
        )

        predictions = [pred_tensor[1:]]
        labels = [label_tensor[1:]]
        lipid_name_dict = {"PC 16:0_18:1": 0, "PE 16:0_18:1": 1, "noise_spectrum": 2}

        accuracy_sum, nr_correct, total, _ = evaluator.evaluate_custom_transformer_accuracy(
            predictions, labels, lipid_name_dict, is_last_epoch=False
        )

        assert total == 1
        # Different class (PE vs PC) but same fatty acids → 0.75 (3/4 correct)
        assert accuracy_sum == 0.75

    def test_last_epoch_updates_confusion_matrix_correct(self, lipid_library):
        """is_last_epoch=True should update confusion matrix for correct predictions."""
        evaluator = Evaluator(lipid_library)

        token_tensor, _ = lipid_library.get_transformer_label(
            "PC 16:0_18:1", "[M+H]+", output_seq_length=15
        )

        predictions = [token_tensor[1:]]
        labels = [token_tensor[1:]]
        lipid_name_dict = {"PC 16:0_18:1": 0, "noise_spectrum": 1}

        _, _, _, matrix = evaluator.evaluate_custom_transformer_accuracy(
            predictions, labels, lipid_name_dict, is_last_epoch=True
        )

        # Correct prediction should increment diagonal
        assert matrix[0, 0] == 1

    def test_last_epoch_updates_confusion_matrix_wrong_prediction(self, lipid_library):
        """is_last_epoch=True should update confusion matrix for wrong predictions."""
        evaluator = Evaluator(lipid_library)

        pred_tensor, _ = lipid_library.get_transformer_label(
            "PE 16:0_18:1", "[M+H]+", output_seq_length=15
        )
        label_tensor, _ = lipid_library.get_transformer_label(
            "PC 16:0_18:1", "[M+H]+", output_seq_length=15
        )

        predictions = [pred_tensor[1:]]
        labels = [label_tensor[1:]]
        lipid_name_dict = {"PC 16:0_18:1": 0, "PE 16:0_18:1": 1, "noise_spectrum": 2}

        _, _, _, matrix = evaluator.evaluate_custom_transformer_accuracy(
            predictions, labels, lipid_name_dict, is_last_epoch=True
        )

        # PE predicted for PC label: matrix[pred_idx=1, label_idx=0]
        assert matrix[1, 0] == 1

    def test_last_epoch_unknown_prediction(self, lipid_library):
        """is_last_epoch=True with unknown prediction should use 'Other' row."""
        evaluator = Evaluator(lipid_library)

        # Create a prediction for a lipid not in the dict
        pred_tensor, _ = lipid_library.get_transformer_label(
            "TG 16:0_18:1_18:2", "[M+H]+", output_seq_length=15
        )
        label_tensor, _ = lipid_library.get_transformer_label(
            "PC 16:0_18:1", "[M+H]+", output_seq_length=15
        )

        predictions = [pred_tensor[1:]]
        labels = [label_tensor[1:]]
        lipid_name_dict = {"PC 16:0_18:1": 0, "noise_spectrum": 1}

        _, _, _, matrix = evaluator.evaluate_custom_transformer_accuracy(
            predictions, labels, lipid_name_dict, is_last_epoch=True
        )

        # Unknown prediction goes to "Other" row (last row = nr_lipids)
        nr_lipids = len(lipid_name_dict)
        assert matrix[nr_lipids, 0] == 1

    def test_last_epoch_noise_spectrum_correct(self, lipid_library):
        """is_last_epoch=True with noise spectrum label and correct prediction."""
        evaluator = Evaluator(lipid_library)

        # Create noise_spectrum tokens (empty lipid)
        pred_tensor, _ = lipid_library.get_transformer_label("", "[M+H]+", output_seq_length=15)

        predictions = [pred_tensor[1:]]
        labels = [pred_tensor[1:]]  # Same as prediction
        lipid_name_dict = {"PC 16:0_18:1": 0, "noise_spectrum": 1}

        _, _, _, matrix = evaluator.evaluate_custom_transformer_accuracy(
            predictions, labels, lipid_name_dict, is_last_epoch=True
        )

        # Noise correctly predicted should go to noise diagonal
        assert matrix[1, 1] == 1

    def test_last_epoch_noise_label_wrong_known_prediction(self, lipid_library):
        """is_last_epoch=True with noise label but known lipid prediction."""
        evaluator = Evaluator(lipid_library)

        pred_tensor, _ = lipid_library.get_transformer_label(
            "PC 16:0_18:1", "[M+H]+", output_seq_length=15
        )
        label_tensor, _ = lipid_library.get_transformer_label("", "[M+H]+", output_seq_length=15)

        predictions = [pred_tensor[1:]]
        labels = [label_tensor[1:]]
        lipid_name_dict = {"PC 16:0_18:1": 0, "noise_spectrum": 1}

        _, _, _, matrix = evaluator.evaluate_custom_transformer_accuracy(
            predictions, labels, lipid_name_dict, is_last_epoch=True
        )

        # PC predicted for noise: matrix[pred_idx=0, noise_idx=1]
        assert matrix[0, 1] == 1

    def test_last_epoch_noise_label_unknown_prediction(self, lipid_library):
        """is_last_epoch=True with noise label and unknown prediction."""
        evaluator = Evaluator(lipid_library)

        pred_tensor, _ = lipid_library.get_transformer_label(
            "TG 16:0_18:1_18:2", "[M+H]+", output_seq_length=15
        )
        label_tensor, _ = lipid_library.get_transformer_label("", "[M+H]+", output_seq_length=15)

        predictions = [pred_tensor[1:]]
        labels = [label_tensor[1:]]
        lipid_name_dict = {"PC 16:0_18:1": 0, "noise_spectrum": 1}

        _, _, _, matrix = evaluator.evaluate_custom_transformer_accuracy(
            predictions, labels, lipid_name_dict, is_last_epoch=True
        )

        # Unknown prediction for noise: matrix[nr_lipids, noise_idx=1]
        nr_lipids = len(lipid_name_dict)
        assert matrix[nr_lipids, 1] == 1


class TestEvaluatorGeneratePredictionInfo:
    """Tests for generate_prediction_info method."""

    def test_generates_prediction_info_dict(self, lipid_library):
        """Should generate complete prediction info dictionary."""
        evaluator = Evaluator(lipid_library)

        # Get actual components from library
        sample_lipid = list(lipid_library.molecular_lipid_species.keys())[0]
        lipid_data = lipid_library.molecular_lipid_species[sample_lipid]

        hg = lipid_data["headgroup"]
        fa1 = lipid_data["fatty_acid_sn1"]
        fa2 = lipid_data["fatty_acid_sn2"]

        info = evaluator.generate_prediction_info(
            label_hg=hg,
            label_sc1=fa1,
            label_sc2=fa2,
            pred_hg=hg,
            pred_sc1=fa1,
            pred_sc2=fa2,
            pred_hg_value=lipid_library.headgroups_mass_norm[hg],
            pred_sc1_value=lipid_library.side_chains_mass_norm[fa1],
            pred_sc2_value=lipid_library.side_chains_mass_norm[fa2],
            batch=0,
            epoch=1,
            idx=0,
        )

        assert info["epoch"] == 1
        assert info["batch"] == 0
        assert info["pred_hg"] == hg
        assert info["label_hg"] == hg


class TestCustomAccuracyUpdate:
    """Tests for CustomAccuracy update method."""

    def test_update_regression_model(self, lipid_library):
        """Should update states for regression model."""
        evaluator = Evaluator(lipid_library)
        lipid_names = ["noise_spectrum"]

        metric = CustomAccuracy(evaluator, lipid_names)

        # Get actual lipid components
        sample_lipid = list(lipid_library.molecular_lipid_species.keys())[0]
        lipid_data = lipid_library.molecular_lipid_species[sample_lipid]

        hg = lipid_data["headgroup"]
        fa1 = lipid_data["fatty_acid_sn1"]
        fa2 = lipid_data["fatty_acid_sn2"]

        predictions = [
            torch.tensor(
                [
                    [lipid_library.headgroups_mass_norm[hg]],
                    [lipid_library.side_chains_mass_norm[fa1]],
                    [lipid_library.side_chains_mass_norm[fa2]],
                ]
            ).T
        ]

        lipid_info = {
            "headgroup": [hg],
            "fatty_acid_sn1": [fa1],
            "fatty_acid_sn2": [fa2],
        }

        metric.update(predictions, lipid_info, model="regression")

        assert metric.total.item() == 1
        assert metric.correct.item() == 1

    def test_update_transformer_model(self, lipid_library):
        """Should update states for transformer model."""
        evaluator = Evaluator(lipid_library)
        lipid_names = ["PC 16:0_18:1", "noise_spectrum"]

        metric = CustomAccuracy(evaluator, lipid_names)

        # Create identical prediction and label
        token_tensor, _ = lipid_library.get_transformer_label(
            "PC 16:0_18:1", "[M+H]+", output_seq_length=15
        )

        predictions = [token_tensor[1:]]  # Skip SOS
        labels = [token_tensor[1:]]

        metric.update(predictions, labels, model="transformer", is_last_epoch=False)

        assert metric.total.item() == 1
        assert metric.correct.item() == 1


class TestCustomLoggerInitialization:
    """Tests for CustomLogger initialization."""

    def test_initialization_transformer(self, tmp_path):
        """CustomLogger should initialize for transformer model."""
        config = {
            "model": "transformer",
            "transformer": {"output_seq_length": 15},
        }
        logger = CustomLogger(
            save_dir=str(tmp_path),
            version="fold_0",
            log_every_n_steps=10,
            config=config,
            do_training=True,
            do_validation=True,
            do_testing=False,
        )

        assert logger.model == "transformer"
        assert logger.output_seq_length == 14  # 15 - 1
        assert logger.do_training is True
        assert logger.do_validation is True
        assert os.path.exists(logger.save_path)

    def test_initialization_regression(self, tmp_path):
        """CustomLogger should initialize for regression model."""
        config = {"model": "regression"}
        logger = CustomLogger(
            save_dir=str(tmp_path),
            version="fold_1",
            log_every_n_steps=5,
            config=config,
            do_training=True,
            do_validation=False,
            do_testing=True,
        )

        assert logger.model == "regression"
        assert logger.do_testing is True
        assert os.path.exists(logger.train_csv_path)
        assert os.path.exists(logger.test_csv_path)

    def test_initialization_no_fold(self, tmp_path):
        """CustomLogger should handle no fold (version='.')."""
        config = {"model": "regression"}
        logger = CustomLogger(
            save_dir=str(tmp_path),
            version=".",
            log_every_n_steps=10,
            config=config,
            do_training=True,
        )

        assert logger.save_path == os.path.join(str(tmp_path), "custom_logger")


class TestCustomLoggerProperties:
    """Tests for CustomLogger properties."""

    def test_name_property(self, tmp_path):
        """name property should return 'custom_logger'."""
        config = {"model": "regression"}
        logger = CustomLogger(
            save_dir=str(tmp_path),
            version=".",
            log_every_n_steps=10,
            config=config,
        )
        assert logger.name == "custom_logger"

    def test_experiment_property(self, tmp_path):
        """experiment property should return None."""
        config = {"model": "regression"}
        logger = CustomLogger(
            save_dir=str(tmp_path),
            version=".",
            log_every_n_steps=10,
            config=config,
        )
        assert logger.experiment is None

    def test_save_dir_property(self, tmp_path):
        """save_dir property should return output directory."""
        config = {"model": "regression"}
        logger = CustomLogger(
            save_dir=str(tmp_path),
            version=".",
            log_every_n_steps=10,
            config=config,
        )
        assert logger.save_dir == str(tmp_path)

    def test_version_property(self, tmp_path):
        """version property should return fold."""
        config = {"model": "regression"}
        logger = CustomLogger(
            save_dir=str(tmp_path),
            version="fold_2",
            log_every_n_steps=10,
            config=config,
        )
        assert logger.version == "fold_2"


class TestCustomLoggerLogHyperparams:
    """Tests for log_hyperparams method."""

    def test_log_hyperparams_does_not_raise(self, tmp_path):
        """log_hyperparams should execute without error."""
        config = {"model": "regression"}
        logger = CustomLogger(
            save_dir=str(tmp_path),
            version=".",
            log_every_n_steps=10,
            config=config,
        )
        # Should not raise
        logger.log_hyperparams({"learning_rate": 0.001})


class TestCustomLoggerLogMetrics:
    """Tests for log_metrics method."""

    def test_log_train_metrics_transformer(self, tmp_path):
        """Should log training metrics for transformer model."""
        config = {
            "model": "transformer",
            "transformer": {"output_seq_length": 15},
        }
        logger = CustomLogger(
            save_dir=str(tmp_path),
            version=".",
            log_every_n_steps=10,
            config=config,
            do_training=True,
        )

        metrics = {
            "epoch": 1,
            "train_loss_epoch": 0.5,
            "customaccuracy_train_accuracy_epoch": 0.85,
            "customaccuracy_train_mean_accuracy_epoch": 0.82,
        }
        logger.log_metrics(metrics, step=0)

        # Verify CSV was written
        import csv

        with open(logger.train_csv_path) as f:
            reader = csv.reader(f)
            rows = list(reader)
            assert len(rows) == 2  # header + 1 data row
            assert rows[1][0] == "1"  # epoch

    def test_log_val_metrics_transformer(self, tmp_path):
        """Should log validation metrics for transformer model."""
        config = {
            "model": "transformer",
            "transformer": {"output_seq_length": 15},
        }
        logger = CustomLogger(
            save_dir=str(tmp_path),
            version=".",
            log_every_n_steps=10,
            config=config,
            do_validation=True,
        )

        metrics = {
            "epoch": 2,
            "val_loss_epoch": 0.4,
            "customaccuracy_val_accuracy_epoch": 0.90,
            "customaccuracy_val_mean_accuracy_epoch": 0.88,
        }
        logger.log_metrics(metrics, step=0)

        import csv

        with open(logger.val_csv_path) as f:
            reader = csv.reader(f)
            rows = list(reader)
            assert len(rows) == 2

    def test_log_test_metrics_transformer(self, tmp_path):
        """Should log test metrics for transformer model."""
        config = {
            "model": "transformer",
            "transformer": {"output_seq_length": 15},
        }
        logger = CustomLogger(
            save_dir=str(tmp_path),
            version=".",
            log_every_n_steps=10,
            config=config,
            do_testing=True,
        )

        metrics = {
            "test_loss_step": 0.3,
            "customaccuracy_test_accuracy_step": 0.92,
            "customaccuracy_test_mean_accuracy_step": 0.90,
        }
        logger.log_metrics(metrics, step=100)

        import csv

        with open(logger.test_csv_path) as f:
            reader = csv.reader(f)
            rows = list(reader)
            assert len(rows) == 2
            assert rows[1][0] == "100"  # step

    def test_log_train_metrics_regression(self, tmp_path):
        """Should log training metrics for regression model."""
        config = {"model": "regression"}
        logger = CustomLogger(
            save_dir=str(tmp_path),
            version=".",
            log_every_n_steps=10,
            config=config,
            do_training=True,
        )

        metrics = {
            "epoch": 1,
            "train_loss_epoch": 0.5,
            "customaccuracy_train_accuracy_epoch": 0.75,
            "train_r2_epoch": 0.85,
            "train_mae_hg_epoch": 0.1,
            "train_mae_fa1_epoch": 0.2,
            "train_mae_fa2_epoch": 0.15,
        }
        logger.log_metrics(metrics, step=0)

        import csv

        with open(logger.train_csv_path) as f:
            reader = csv.reader(f)
            rows = list(reader)
            assert len(rows) == 2

    def test_log_val_metrics_regression(self, tmp_path):
        """Should log validation metrics for regression model."""
        config = {"model": "regression"}
        logger = CustomLogger(
            save_dir=str(tmp_path),
            version=".",
            log_every_n_steps=10,
            config=config,
            do_validation=True,
        )

        metrics = {
            "epoch": 1,
            "val_loss_epoch": 0.4,
            "customaccuracy_val_accuracy_epoch": 0.80,
            "val_mae_hg_epoch": 0.08,
            "val_mae_fa1_epoch": 0.18,
            "val_mae_fa2_epoch": 0.12,
        }
        logger.log_metrics(metrics, step=0)

        import csv

        with open(logger.val_csv_path) as f:
            reader = csv.reader(f)
            rows = list(reader)
            assert len(rows) == 2

    def test_log_test_metrics_regression(self, tmp_path):
        """Should log test metrics for regression model."""
        config = {"model": "regression"}
        logger = CustomLogger(
            save_dir=str(tmp_path),
            version=".",
            log_every_n_steps=10,
            config=config,
            do_testing=True,
        )

        metrics = {
            "test_loss_step": 0.35,
            "customaccuracy_test_accuracy_step": 0.82,
            "test_mae_hg_step": 0.07,
            "test_mae_fa1_step": 0.15,
            "test_mae_fa2_step": 0.10,
        }
        logger.log_metrics(metrics, step=50)


class TestCustomLoggerSave:
    """Tests for save method."""

    def test_save_does_not_raise(self, tmp_path):
        """save method should execute without error."""
        config = {"model": "regression"}
        logger = CustomLogger(
            save_dir=str(tmp_path),
            version=".",
            log_every_n_steps=10,
            config=config,
        )
        # Should not raise
        logger.save()


class TestCustomAccuracyUpdateWithSync:
    """Tests for CustomAccuracy update with sync behavior."""

    def test_update_transformer_with_last_epoch_and_sync(self, lipid_library):
        """Should update confusion matrix when is_last_epoch=True and _to_sync=True."""
        evaluator = Evaluator(lipid_library)
        lipid_names = ["PC 16:0_18:1", "noise_spectrum"]

        metric = CustomAccuracy(evaluator, lipid_names)
        # Enable sync mode (line 176 coverage)
        metric._to_sync = True

        token_tensor, _ = lipid_library.get_transformer_label(
            "PC 16:0_18:1", "[M+H]+", output_seq_length=15
        )

        predictions = [token_tensor[1:]]
        labels = [token_tensor[1:]]

        initial_matrix_sum = metric.confusion_matrix.sum().item()
        metric.update(predictions, labels, model="transformer", is_last_epoch=True)

        # Confusion matrix should be updated
        assert metric.confusion_matrix.sum().item() > initial_matrix_sum


class TestCustomLoggerFinalize:
    """Tests for finalize method."""

    def test_finalize_training_transformer(self, tmp_path):
        """finalize should create training plots for transformer."""
        import csv

        config = {
            "model": "transformer",
            "transformer": {"output_seq_length": 15},
        }
        logger = CustomLogger(
            save_dir=str(tmp_path),
            version=".",
            log_every_n_steps=10,
            config=config,
            do_training=True,
        )

        # Write some data to the CSV
        with open(logger.train_csv_path, "a") as f:
            writer = csv.writer(f)
            writer.writerow([1, 0.5, "85.00", "82.00"])
            writer.writerow([2, 0.4, "90.00", "88.00"])

        # finalize should run without error
        logger.finalize(status="success")

        # Transformer training only generates the loss plot (accuracy tracking is disabled
        # for performance; see training_step in lightning_module.py)
        assert os.path.exists(os.path.join(logger.save_path, "plot_loss_training.png"))

    def test_finalize_validation_transformer(self, tmp_path):
        """finalize should create validation plots for transformer."""
        import csv

        config = {
            "model": "transformer",
            "transformer": {"output_seq_length": 15},
        }
        logger = CustomLogger(
            save_dir=str(tmp_path),
            version=".",
            log_every_n_steps=10,
            config=config,
            do_validation=True,
        )

        with open(logger.val_csv_path, "a") as f:
            writer = csv.writer(f)
            writer.writerow([1, 0.4, "88.00", "85.00"])

        logger.finalize(status="success")

        assert os.path.exists(os.path.join(logger.save_path, "plot_loss_accuracy_validation.png"))

    def test_finalize_training_regression(self, tmp_path):
        """finalize should create training plots for regression."""
        import csv

        config = {"model": "regression"}
        logger = CustomLogger(
            save_dir=str(tmp_path),
            version=".",
            log_every_n_steps=10,
            config=config,
            do_training=True,
        )

        with open(logger.train_csv_path, "a") as f:
            writer = csv.writer(f)
            writer.writerow([1, 0.5, 0.75, 0.85, 0.1, 0.2, 0.15])

        logger.finalize(status="success")

        assert os.path.exists(os.path.join(logger.save_path, "plot_loss_accuracy_training.png"))
        assert os.path.exists(os.path.join(logger.save_path, "plot_loss_mae_training.png"))
        assert os.path.exists(os.path.join(logger.save_path, "plot_loss_r2_training.png"))

    def test_finalize_validation_regression(self, tmp_path):
        """finalize should create validation plots for regression."""
        import csv

        config = {"model": "regression"}
        logger = CustomLogger(
            save_dir=str(tmp_path),
            version=".",
            log_every_n_steps=10,
            config=config,
            do_validation=True,
        )

        with open(logger.val_csv_path, "a") as f:
            writer = csv.writer(f)
            writer.writerow([1, 0.4, 0.80, 0.88, 0.08, 0.18, 0.12])

        logger.finalize(status="success")

        assert os.path.exists(os.path.join(logger.save_path, "plot_loss_accuracy_validation.png"))


class TestCustomLoggerSaveLipidWiseMetrics:
    """Tests for save_lipid_wise_metrics method."""

    def test_save_train_confusion_matrix(self, tmp_path):
        """Should save training confusion matrix."""
        config = {"model": "transformer", "transformer": {"output_seq_length": 15}}
        logger = CustomLogger(
            save_dir=str(tmp_path),
            version=".",
            log_every_n_steps=10,
            config=config,
            do_training=True,
            trainset_lipids=["PC 16:0_18:1", "PE 18:0_18:2"],
        )

        train_matrix = torch.tensor([[10.0, 2.0], [1.0, 8.0], [0.0, 1.0]])

        logger.save_lipid_wise_metrics(
            train_confusion_matrix=train_matrix,
            train_lipids=["PC 16:0_18:1", "PE 18:0_18:2"],
        )

        assert os.path.exists(os.path.join(logger.save_path, "confusion_matrix_train.csv"))

    def test_save_val_confusion_matrix(self, tmp_path):
        """Should save validation confusion matrix."""
        config = {"model": "transformer", "transformer": {"output_seq_length": 15}}
        logger = CustomLogger(
            save_dir=str(tmp_path),
            version=".",
            log_every_n_steps=10,
            config=config,
            do_validation=True,
            valset_lipids=["PC 16:0_18:1"],
        )

        val_matrix = torch.tensor([[15.0], [2.0]])

        logger.save_lipid_wise_metrics(
            val_confusion_matrix=val_matrix,
            val_lipids=["PC 16:0_18:1"],
        )

        assert os.path.exists(os.path.join(logger.save_path, "confusion_matrix_val.csv"))

    def test_save_test_confusion_matrix(self, tmp_path):
        """Should save test confusion matrix."""
        config = {"model": "transformer", "transformer": {"output_seq_length": 15}}
        logger = CustomLogger(
            save_dir=str(tmp_path),
            version=".",
            log_every_n_steps=10,
            config=config,
            do_testing=True,
            testset_lipids=["PC 16:0_18:1", "PE 18:0_18:2"],
        )

        test_matrix = torch.tensor([[20.0, 3.0], [2.0, 18.0], [1.0, 0.0]])

        logger.save_lipid_wise_metrics(
            test_confusion_matrix=test_matrix,
            test_lipids=["PC 16:0_18:1", "PE 18:0_18:2"],
        )

        assert os.path.exists(os.path.join(logger.save_path, "confusion_matrix_test.csv"))


class TestTransformTokenPredictionsToString:
    """Tests for transform_token_predictions_to_string method."""

    def test_transforms_correct_prediction(self, tmp_path, lipid_library):
        """Should transform matching prediction and label correctly."""
        output_seq_length = 15
        config = {
            "model": "transformer",
            "transformer": {"output_seq_length": output_seq_length},
        }
        logger = CustomLogger(
            save_dir=str(tmp_path),
            version=".",
            log_every_n_steps=10,
            config=config,
        )

        # Generate tokens for a lipid
        token_tensor, _ = lipid_library.get_transformer_label(
            "PC 16:0_18:1", "[M+H]+", output_seq_length=output_seq_length
        )
        # Skip SOS token (index 0), use indices 1 onwards
        pred_tokens = token_tensor[1:].numpy()
        label_tokens = token_tensor[1:].numpy()

        # Build input array: [epoch, batch, pred_tokens..., label_tokens..., dataset_idx]
        epoch = 1
        batch = 0
        dataset_idx = 0
        dataset_names = ["path/to/dataset.hdf5"]

        row = np.concatenate([[epoch, batch], pred_tokens, label_tokens, [dataset_idx]])
        preds_vs_labels = np.array([row])

        result = logger.transform_token_predictions_to_string(preds_vs_labels, dataset_names)

        assert result.shape[0] == 1
        assert str(result[0][0]) == str(epoch)  # epoch
        assert str(result[0][1]) == str(batch)  # batch
        assert "PC " in str(result[0][2])  # prediction contains lipid class
        assert "16:0" in str(result[0][2])  # prediction contains fatty acid
        assert result[0][2] == result[0][3]  # prediction == label
        assert result[0][4] == dataset_names[0]  # dataset path
        assert float(result[0][5]) == 1.0  # custom_accuracy
        assert str(result[0][6]) == "True"  # custom_correct
        assert str(result[0][7]) == "True"  # correct

    def test_transforms_wrong_prediction(self, tmp_path, lipid_library):
        """Should transform mismatched prediction and label correctly."""
        output_seq_length = 15
        config = {
            "model": "transformer",
            "transformer": {"output_seq_length": output_seq_length},
        }
        logger = CustomLogger(
            save_dir=str(tmp_path),
            version=".",
            log_every_n_steps=10,
            config=config,
        )

        # Generate tokens for different lipids
        pred_tensor, _ = lipid_library.get_transformer_label(
            "PE 16:0_18:1", "[M+H]+", output_seq_length=output_seq_length
        )
        label_tensor, _ = lipid_library.get_transformer_label(
            "PC 16:0_18:1", "[M+H]+", output_seq_length=output_seq_length
        )

        pred_tokens = pred_tensor[1:].numpy()
        label_tokens = label_tensor[1:].numpy()

        epoch = 2
        batch = 5
        dataset_idx = 0
        dataset_names = ["train_data.hdf5"]

        row = np.concatenate([[epoch, batch], pred_tokens, label_tokens, [dataset_idx]])
        preds_vs_labels = np.array([row])

        result = logger.transform_token_predictions_to_string(preds_vs_labels, dataset_names)

        assert "PE " in str(result[0][2])  # prediction is PE
        assert "PC " in str(result[0][3])  # label is PC
        assert result[0][2] != result[0][3]  # prediction != label
        assert float(result[0][5]) == 0.75  # custom_accuracy (same FA, different class)
        assert str(result[0][6]) == "False"  # custom_correct
        assert str(result[0][7]) == "False"  # correct

    def test_transforms_multiple_rows(self, tmp_path, lipid_library):
        """Should transform multiple predictions correctly."""
        output_seq_length = 15
        config = {
            "model": "transformer",
            "transformer": {"output_seq_length": output_seq_length},
        }
        logger = CustomLogger(
            save_dir=str(tmp_path),
            version=".",
            log_every_n_steps=10,
            config=config,
        )

        token_tensor, _ = lipid_library.get_transformer_label(
            "PC 16:0_18:1", "[M+H]+", output_seq_length=output_seq_length
        )
        tokens = token_tensor[1:].numpy()

        dataset_names = ["dataset1.hdf5", "dataset2.hdf5"]

        rows = []
        for i in range(3):
            row = np.concatenate([[i, i * 10], tokens, tokens, [i % 2]])
            rows.append(row)
        preds_vs_labels = np.array(rows)

        result = logger.transform_token_predictions_to_string(preds_vs_labels, dataset_names)

        assert result.shape[0] == 3
        assert str(result[0][0]) == "0"  # first row epoch
        assert str(result[1][0]) == "1"  # second row epoch
        assert str(result[2][0]) == "2"  # third row epoch


class TestTransformTestTokenPredictionsToString:
    """Tests for transform_test_token_predictions_to_string method."""

    def test_transforms_test_prediction_correct(self, tmp_path, lipid_library):
        """Should transform matching test prediction and label correctly."""
        output_seq_length = 15
        config = {
            "model": "transformer",
            "transformer": {"output_seq_length": output_seq_length},
        }
        logger = CustomLogger(
            save_dir=str(tmp_path),
            version=".",
            log_every_n_steps=10,
            config=config,
        )

        token_tensor, _ = lipid_library.get_transformer_label(
            "PC 16:0_18:1", "[M+H]+", output_seq_length=output_seq_length
        )
        pred_tokens = token_tensor[1:].numpy()
        label_tokens = token_tensor[1:].numpy()

        # Test format: [pred_tokens..., label_tokens..., dataset_idx, confidence_score]
        dataset_idx = 0
        confidence_score = 0.95
        dataset_names = ["test_data.hdf5"]

        row = np.concatenate([pred_tokens, label_tokens, [dataset_idx], [confidence_score]])
        preds_vs_labels = np.array([row])

        result = logger.transform_test_token_predictions_to_string(preds_vs_labels, dataset_names)

        assert result.shape[0] == 1
        assert "PC " in str(result[0][0])  # prediction
        assert "PC " in str(result[0][1])  # label
        assert result[0][0] == result[0][1]  # prediction == label
        assert result[0][2] == dataset_names[0]  # dataset path
        assert float(result[0][3]) == 1.0  # custom_accuracy
        assert str(result[0][4]) == "True"  # custom_correct
        assert str(result[0][5]) == "True"  # correct
        assert float(result[0][6]) == confidence_score  # confidence_score

    def test_transforms_test_prediction_wrong(self, tmp_path, lipid_library):
        """Should transform mismatched test prediction correctly."""
        output_seq_length = 15
        config = {
            "model": "transformer",
            "transformer": {"output_seq_length": output_seq_length},
        }
        logger = CustomLogger(
            save_dir=str(tmp_path),
            version=".",
            log_every_n_steps=10,
            config=config,
        )

        pred_tensor, _ = lipid_library.get_transformer_label(
            "PE 18:0_18:2", "[M+H]+", output_seq_length=output_seq_length
        )
        label_tensor, _ = lipid_library.get_transformer_label(
            "PC 16:0_18:1", "[M+H]+", output_seq_length=output_seq_length
        )

        pred_tokens = pred_tensor[1:].numpy()
        label_tokens = label_tensor[1:].numpy()

        dataset_idx = 0
        confidence_score = 0.72
        dataset_names = ["test_set.hdf5"]

        row = np.concatenate([pred_tokens, label_tokens, [dataset_idx], [confidence_score]])
        preds_vs_labels = np.array([row])

        result = logger.transform_test_token_predictions_to_string(preds_vs_labels, dataset_names)

        assert "PE " in str(result[0][0])  # prediction is PE
        assert "PC " in str(result[0][1])  # label is PC
        assert result[0][0] != result[0][1]  # prediction != label
        assert float(result[0][3]) < 1.0  # custom_accuracy < 1
        assert str(result[0][4]) == "False"  # custom_correct
        assert str(result[0][5]) == "False"  # correct
        assert float(result[0][6]) == confidence_score

    def test_transforms_test_multiple_rows(self, tmp_path, lipid_library):
        """Should transform multiple test predictions correctly."""
        output_seq_length = 15
        config = {
            "model": "transformer",
            "transformer": {"output_seq_length": output_seq_length},
        }
        logger = CustomLogger(
            save_dir=str(tmp_path),
            version=".",
            log_every_n_steps=10,
            config=config,
        )

        token_tensor, _ = lipid_library.get_transformer_label(
            "PC 16:0_18:1", "[M+H]+", output_seq_length=output_seq_length
        )
        tokens = token_tensor[1:].numpy()

        dataset_names = ["test1.hdf5", "test2.hdf5"]

        rows = []
        for i in range(4):
            confidence = 0.8 + i * 0.05
            row = np.concatenate([tokens, tokens, [i % 2], [confidence]])
            rows.append(row)
        preds_vs_labels = np.array(rows)

        result = logger.transform_test_token_predictions_to_string(preds_vs_labels, dataset_names)

        assert result.shape[0] == 4
        # All predictions should be correct since pred == label
        for i in range(4):
            assert str(result[i][4]) == "True"  # custom_correct
            assert str(result[i][5]) == "True"  # correct


class TestPredictionLoggerInitialization:
    """Tests for PredictionLogger initialization."""

    def test_initialization_with_minimal_config(self, tmp_path):
        """Should initialize with minimal config."""
        from lipidetective.helpers.logging import PredictionLogger

        config = {
            "model": "transformer",
            "transformer": {"output_seq_length": 11},
        }

        logger = PredictionLogger(
            save_dir=str(tmp_path),
            log_every_n_steps=1,
            config=config,
        )

        assert logger.output_directory == str(tmp_path)
        assert logger.log_every_n_steps == 1
        assert logger.model == "transformer"
        assert logger.output_seq_length == 10  # 11 - 1

    def test_initialization_with_predict_config(self, tmp_path):
        """Should parse predict config options."""
        from lipidetective.helpers.logging import PredictionLogger

        config = {
            "model": "transformer",
            "transformer": {"output_seq_length": 11},
            "predict": {
                "output": "top3",
                "save_spectrum": True,
                "keep_empty": True,
                "keep_wrong_polarity_preds": True,
                "confidence_threshold": 0.5,
            },
        }

        logger = PredictionLogger(
            save_dir=str(tmp_path),
            log_every_n_steps=1,
            config=config,
        )

        assert logger.save_top3 is True
        assert logger.save_spectrum is True
        assert logger.keep_empty is True
        assert logger.keep_wrong_polarity_preds is True
        assert logger.confidence_threshold == 0.5

    def test_initialization_defaults_without_predict_section(self, tmp_path):
        """Should use defaults when predict section missing."""
        from lipidetective.helpers.logging import PredictionLogger

        config = {
            "model": "transformer",
            "transformer": {"output_seq_length": 11},
        }

        logger = PredictionLogger(
            save_dir=str(tmp_path),
            log_every_n_steps=1,
            config=config,
        )

        assert logger.save_top3 is False
        assert logger.save_spectrum is False
        assert logger.keep_empty is False
        assert logger.keep_wrong_polarity_preds is False
        assert logger.confidence_threshold == 0


class TestPredictionLoggerProperties:
    """Tests for PredictionLogger properties."""

    def test_name_property(self, tmp_path):
        """Should return 'custom_logger'."""
        from lipidetective.helpers.logging import PredictionLogger

        config = {"model": "transformer", "transformer": {"output_seq_length": 11}}
        logger = PredictionLogger(str(tmp_path), 1, config)

        assert logger.name == "custom_logger"

    def test_experiment_property(self, tmp_path):
        """Should return None."""
        from lipidetective.helpers.logging import PredictionLogger

        config = {"model": "transformer", "transformer": {"output_seq_length": 11}}
        logger = PredictionLogger(str(tmp_path), 1, config)

        assert logger.experiment is None

    def test_save_dir_property(self, tmp_path):
        """Should return output directory."""
        from lipidetective.helpers.logging import PredictionLogger

        config = {"model": "transformer", "transformer": {"output_seq_length": 11}}
        logger = PredictionLogger(str(tmp_path), 1, config)

        assert logger.save_dir == str(tmp_path)

    def test_version_property(self, tmp_path):
        """Should return '.'."""
        from lipidetective.helpers.logging import PredictionLogger

        config = {"model": "transformer", "transformer": {"output_seq_length": 11}}
        logger = PredictionLogger(str(tmp_path), 1, config)

        assert logger.version == "."


class TestPredictionLoggerGenerateOutputFiles:
    """Tests for generate_output_files method."""

    def test_creates_predictions_csv(self, tmp_path):
        """Should create predictions.csv with header."""
        from lipidetective.helpers.logging import PredictionLogger

        config = {"model": "transformer", "transformer": {"output_seq_length": 11}}
        logger = PredictionLogger(str(tmp_path), 1, config)

        assert logger.predictions_path is not None
        predictions_file = tmp_path / "predictions.csv"
        assert predictions_file.exists()

        with open(predictions_file) as f:
            header = f.readline().strip()
        assert "file" in header
        assert "polarity" in header
        assert "prediction" in header
        assert "confidence" in header

    def test_creates_top3_predictions_when_enabled(self, tmp_path):
        """Should create top3_predictions.csv when save_top3 is True."""
        from lipidetective.helpers.logging import PredictionLogger

        config = {
            "model": "transformer",
            "transformer": {"output_seq_length": 11},
            "predict": {"output": "top3"},
        }
        logger = PredictionLogger(str(tmp_path), 1, config)

        assert logger.top_3_predictions_path is not None
        top3_file = tmp_path / "top3_predictions.csv"
        assert top3_file.exists()

        with open(top3_file) as f:
            header = f.readline().strip()
        assert "prediction_1" in header
        assert "prediction_2" in header
        assert "prediction_3" in header

    def test_no_top3_file_when_disabled(self, tmp_path):
        """Should not create top3_predictions.csv when save_top3 is False."""
        from lipidetective.helpers.logging import PredictionLogger

        config = {"model": "transformer", "transformer": {"output_seq_length": 11}}
        logger = PredictionLogger(str(tmp_path), 1, config)

        top3_file = tmp_path / "top3_predictions.csv"
        assert not top3_file.exists()
        assert logger.top_3_predictions_path is None


class TestPredictionLoggerMethods:
    """Tests for PredictionLogger utility methods."""

    def test_log_metrics_does_not_raise(self, tmp_path):
        """log_metrics should not raise (no-op method)."""
        from lipidetective.helpers.logging import PredictionLogger

        config = {"model": "transformer", "transformer": {"output_seq_length": 11}}
        logger = PredictionLogger(str(tmp_path), 1, config)

        # Should not raise
        logger.log_metrics({"loss": 0.5}, step=1)

    def test_log_hyperparams_does_not_raise(self, tmp_path):
        """log_hyperparams should not raise (no-op method)."""
        from lipidetective.helpers.logging import PredictionLogger

        config = {"model": "transformer", "transformer": {"output_seq_length": 11}}
        logger = PredictionLogger(str(tmp_path), 1, config)

        # Should not raise
        logger.log_hyperparams({"lr": 0.001})

    def test_save_does_not_raise(self, tmp_path):
        """save should not raise (no-op method)."""
        from lipidetective.helpers.logging import PredictionLogger

        config = {"model": "transformer", "transformer": {"output_seq_length": 11}}
        logger = PredictionLogger(str(tmp_path), 1, config)

        # Should not raise
        logger.save()

    def test_finalize_does_not_raise(self, tmp_path):
        """finalize should not raise (no-op method)."""
        from lipidetective.helpers.logging import PredictionLogger

        config = {"model": "transformer", "transformer": {"output_seq_length": 11}}
        logger = PredictionLogger(str(tmp_path), 1, config)

        # Should not raise
        logger.finalize("success")
