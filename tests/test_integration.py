"""Integration tests that run complete LipiDetective workflows end-to-end.

These tests use a real 2-spectrum HDF5 dataset and exercise the full pipeline:
Trainer initialization -> data loading -> model training/testing/prediction.

All tests are marked ``slow`` so ``make test-fast`` skips them.
"""

import os

import pytest

from lipidetective.workflow.trainer import Trainer

pytestmark = [pytest.mark.slow, pytest.mark.integration]


# ---------------------------------------------------------------------------
# Training workflows
# ---------------------------------------------------------------------------


class TestTrainWithoutValidation:
    """Train a transformer for 1 epoch without validation."""

    def test_completes_and_creates_output(self, integration_base_config):
        config = integration_base_config.copy()
        config["workflow"] = {**config["workflow"], "train": True, "save_model": True}

        trainer = Trainer(config)
        trainer.train_without_validation()

        # Trainer creates a timestamped subfolder inside config output dir
        output_contents = os.listdir(config["files"]["output"])
        output_dirs = [
            d
            for d in output_contents
            if os.path.isdir(os.path.join(config["files"]["output"], d))
            and d.startswith("LipiDetective_Output_")
        ]
        assert len(output_dirs) == 1, f"Expected 1 output dir, found: {output_dirs}"

        output_folder = os.path.join(config["files"]["output"], output_dirs[0])
        saved_model = os.path.join(output_folder, "lipidetective_model.pth")
        assert os.path.isfile(saved_model), "Model file was not saved"


class TestTrainWithValidation:
    """Train a transformer with k-fold cross-validation (k=2, 2 lipid species)."""

    def test_kfold_completes(self, integration_base_config):
        config = integration_base_config.copy()
        config["workflow"] = {
            **config["workflow"],
            "train": True,
            "validate": True,
            "save_model": False,
        }

        trainer = Trainer(config)
        trainer.train_with_validation()

        output_contents = os.listdir(config["files"]["output"])
        output_dirs = [
            d
            for d in output_contents
            if os.path.isdir(os.path.join(config["files"]["output"], d))
            and d.startswith("LipiDetective_Output_")
        ]
        assert len(output_dirs) == 1


# ---------------------------------------------------------------------------
# Test workflow (requires a pre-trained model)
# ---------------------------------------------------------------------------


class TestTestWorkflow:
    """Load a saved model and evaluate on the HDF5 test set."""

    def test_evaluates_on_test_data(self, integration_base_config, trained_model_artifact):
        model_path, _train_output = trained_model_artifact

        config = integration_base_config.copy()
        config["workflow"] = {
            **config["workflow"],
            "test": True,
            "load_model": True,
        }
        config["files"] = {**config["files"], "saved_model": model_path}

        trainer = Trainer(config)
        trainer.test()


# ---------------------------------------------------------------------------
# Predict workflow (requires a pre-trained model)
# ---------------------------------------------------------------------------


class TestPredictWorkflow:
    """Load a saved model and predict from a JSON file."""

    def test_predicts_from_json(
        self, integration_base_config, trained_model_artifact, prediction_json_path
    ):
        model_path, _train_output = trained_model_artifact

        config = integration_base_config.copy()
        config["workflow"] = {
            **config["workflow"],
            "predict": True,
            "load_model": True,
        }
        config["files"] = {
            **config["files"],
            "saved_model": model_path,
            "predict_input": prediction_json_path,
        }

        trainer = Trainer(config)
        trainer.predict()

        # Check that predictions CSV was created in the output folder
        output_contents = os.listdir(config["files"]["output"])
        output_dirs = [
            d
            for d in output_contents
            if os.path.isdir(os.path.join(config["files"]["output"], d))
            and d.startswith("LipiDetective_Output_")
        ]
        assert len(output_dirs) >= 1, "No output directory created"
        output_folder = os.path.join(config["files"]["output"], output_dirs[0])
        assert os.path.isfile(os.path.join(output_folder, "predictions.csv")), (
            "predictions.csv not found"
        )


# ---------------------------------------------------------------------------
# Random forest workflows
# ---------------------------------------------------------------------------


class TestRandomForestWorkflow:
    """Test both random forest data loading modes."""

    def test_leakage_mode(self, integration_hdf5_path, tmp_path):
        """RF with leakage mode: random split from all_datasets/."""
        config = {
            "model": "random_forest",
            "files": {
                "train_input": integration_hdf5_path,
                "output": str(tmp_path),
                "splitting_instructions": "leakage",
            },
            "random_forest": {"type": "single_classifier"},
        }

        rf = Trainer(config)
        rf.run_random_forest()

    def test_species_aware_split(
        self, integration_hdf5_path, splitting_instructions_path, tmp_path
    ):
        """RF with species-aware split from lipid_classes/ using splitting instructions."""
        config = {
            "model": "random_forest",
            "files": {
                "train_input": integration_hdf5_path,
                "output": str(tmp_path),
                "splitting_instructions": splitting_instructions_path,
            },
            "random_forest": {"type": "single_classifier"},
        }

        rf = Trainer(config)
        rf.run_random_forest()
