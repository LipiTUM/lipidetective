"""Unit tests for trainer.py."""

import os
import tempfile

import pytest
from lipidetective.workflow.trainer import (
    CustomLoggerCallback,
    ModifiedASHAScheduler,
    PrintingCallbacks,
    Trainer,
)


class TestParseLipidDatasetName:
    """Tests for parse_lipid_dataset_name method."""

    def test_extracts_lipid_name_before_pipe(self, trainer_instance):
        """Should extract lipid name before ' | ' separator."""
        result = trainer_instance.parse_lipid_dataset_name("PC 16:0_18:1 | dataset_1")

        assert result == "PC 16:0_18:1"

    def test_handles_name_without_pipe(self, trainer_instance):
        """Should return full name when no pipe separator."""
        result = trainer_instance.parse_lipid_dataset_name("PC 16:0_18:1")

        assert result == "PC 16:0_18:1"

    def test_converts_underscore_to_slash_for_ceramides(self, trainer_instance):
        """Should convert underscore to slash for ceramide-family lipids."""
        result = trainer_instance.parse_lipid_dataset_name("Cer 18:1_16:0 | dataset_1")

        assert result == "Cer 18:1/16:0"

    def test_converts_underscore_to_slash_for_sm(self, trainer_instance):
        """Should convert underscore to slash for sphingomyelin."""
        result = trainer_instance.parse_lipid_dataset_name("SM d18:1_16:0 | dataset_1")

        assert result == "SM d18:1/16:0"

    def test_preserves_underscore_for_pc(self, trainer_instance):
        """Should preserve underscore for phosphatidylcholine."""
        result = trainer_instance.parse_lipid_dataset_name("PC 16:0_18:1 | dataset_1")

        assert result == "PC 16:0_18:1"

    def test_preserves_underscore_for_pe(self, trainer_instance):
        """Should preserve underscore for phosphatidylethanolamine."""
        result = trainer_instance.parse_lipid_dataset_name("PE 18:0_18:2 | dataset_1")

        assert result == "PE 18:0_18:2"


class TestGetUniqueLipids:
    """Tests for get_unique_lipids method."""

    def test_returns_unique_lipids(self, trainer_instance):
        """Should return list of unique lipid names."""
        dataset_list = [
            "PC 16:0_18:1 | dataset_1",
            "PC 16:0_18:1 | dataset_2",
            "PE 18:0_18:2 | dataset_1",
        ]

        result = trainer_instance.get_unique_lipids(dataset_list)

        assert len(result) == 2
        assert "PC 16:0_18:1" in result
        assert "PE 18:0_18:2" in result

    def test_handles_empty_list(self, trainer_instance):
        """Should return empty list for empty input."""
        result = trainer_instance.get_unique_lipids([])

        assert result == []

    def test_handles_single_lipid(self, trainer_instance):
        """Should handle single lipid in list."""
        dataset_list = ["PC 34:1 | spectrum_1"]

        result = trainer_instance.get_unique_lipids(dataset_list)

        assert len(result) == 1


class TestGetPredFiles:
    """Tests for get_pred_files method."""

    def test_returns_single_file_as_list(self, trainer_instance):
        """Should return single file path as list."""
        with tempfile.NamedTemporaryFile(suffix=".mzML", delete=False) as f:
            temp_file = f.name

        try:
            trainer_instance.config["files"]["predict_input"] = temp_file
            result = trainer_instance.get_pred_files()

            assert result == [temp_file]
        finally:
            os.unlink(temp_file)

    def test_returns_files_from_directory(self, trainer_instance):
        """Should return all files from directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create test files
            file1 = os.path.join(tmp_dir, "test1.mzML")
            file2 = os.path.join(tmp_dir, "test2.mzML")
            with open(file1, "w") as f:
                f.write("test")
            with open(file2, "w") as f:
                f.write("test")

            trainer_instance.config["files"]["predict_input"] = tmp_dir
            result = trainer_instance.get_pred_files()

            assert len(result) == 2
            assert file1 in result
            assert file2 in result

    def test_excludes_subdirectories(self, trainer_instance):
        """Should not include subdirectories."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            file1 = os.path.join(tmp_dir, "test1.mzML")
            subdir = os.path.join(tmp_dir, "subdir")
            with open(file1, "w") as f:
                f.write("test")
            os.makedirs(subdir)

            trainer_instance.config["files"]["predict_input"] = tmp_dir
            result = trainer_instance.get_pred_files()

            assert len(result) == 1
            assert file1 in result


class TestCheckParameterForTuning:
    """Tests for check_parameter_for_tuning method."""

    def test_converts_list_to_grid_search(self, trainer_instance):
        """Should convert list values to tune.grid_search."""
        trainer_instance.config["training"]["learning_rate"] = [0.001, 0.01, 0.1]

        trainer_instance.check_parameter_for_tuning("training", "learning_rate")

        # After conversion, it should be a grid_search object (not a plain list)
        assert not isinstance(trainer_instance.config["training"]["learning_rate"], list)

    def test_leaves_non_list_unchanged(self, trainer_instance):
        """Should not modify non-list values."""
        trainer_instance.config["training"]["learning_rate"] = 0.001

        trainer_instance.check_parameter_for_tuning("training", "learning_rate")

        assert trainer_instance.config["training"]["learning_rate"] == 0.001


class TestPrintingCallbacks:
    """Tests for PrintingCallbacks class."""

    def test_instantiation(self):
        """Should instantiate without error."""
        callback = PrintingCallbacks()
        assert callback is not None


class TestCustomLoggerCallback:
    """Tests for CustomLoggerCallback class."""

    def test_instantiation(self):
        """Should instantiate with output folder."""
        callback = CustomLoggerCallback("/tmp/output")
        assert callback.output_folder == "/tmp/output"


class TestModifiedASHAScheduler:
    """Tests for ModifiedASHAScheduler class."""

    def test_instantiation(self):
        """Should instantiate with ASHA parameters."""
        scheduler = ModifiedASHAScheduler(max_t=10, grace_period=2, reduction_factor=4)
        assert scheduler is not None


@pytest.fixture
def trainer_instance(monkeypatch):
    """Create a minimal Trainer instance for testing utility methods."""
    # We need to mock several things to create a Trainer without full setup

    # Minimal config for testing
    config = {
        "model": "transformer",
        "workflow": {"log_every_n_steps": 10},
        "files": {
            "output": "/tmp/test_output",
            "train_input": None,
            "val_input": None,
            "predict_input": None,
            "splitting_instructions": None,
        },
        "training": {
            "batch": 4,
            "epochs": 2,
            "learning_rate": 0.001,
            "lr_step": 5,
            "nr_workers": 0,
            "k": 5,
        },
        "cuda": {"gpu_nr": None},
        "input_embedding": {
            "n_peaks": 30,
            "max_mz": 1600,
            "decimal_accuracy": 1,
        },
        "transformer": {
            "d_model": 32,
            "num_heads": 4,
            "dropout": 0.1,
            "ffn_hidden": 64,
            "num_layers": 2,
            "output_seq_length": 11,
        },
    }

    # Mock is_main_process to return False to skip output folder creation
    monkeypatch.setattr("lipidetective.workflow.trainer.is_main_process", lambda: False)

    # Mock torch.cuda.is_available
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    # Create trainer with mocked dependencies
    trainer = Trainer(config)

    return trainer
