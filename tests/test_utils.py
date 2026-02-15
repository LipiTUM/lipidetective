"""Unit tests for helpers/utils.py functions."""

import logging as _logging
import os
import tempfile

import numpy as np
import pytest


class TestReadYaml:
    """Tests for YAML file reading."""

    def test_read_yaml(self, sample_yaml_file):
        """Test YAML file reading."""
        from lipidetective.helpers.utils import read_yaml

        config = read_yaml(sample_yaml_file)

        assert config is not None
        assert config["model"] == "transformer"
        assert config["workflow"]["train"] is True
        assert config["transformer"]["d_model"] == 128

    def test_read_yaml_nonexistent_file(self):
        """Test read_yaml handles missing files gracefully."""
        from lipidetective.helpers.utils import read_yaml

        result = read_yaml("/nonexistent/path/config.yaml")
        assert result is None


class TestTruncate:
    """Tests for numerical truncation."""

    def test_truncate_zero_decimal_places(self):
        """Test truncating to whole numbers."""
        from lipidetective.helpers.utils import truncate

        values = np.array([1.2345, 2.6789, 3.9999])
        result = truncate(values, decimal_places=0)
        assert result == [1.0, 2.0, 3.0]

    def test_truncate_two_decimal_places(self):
        """Test truncating to two decimal places."""
        from lipidetective.helpers.utils import truncate

        values = np.array([1.2345, 2.6789, 3.9999])
        result = truncate(values, decimal_places=2)
        assert result == [1.23, 2.67, 3.99]


class TestLipidClassDetection:
    """Tests for lipid class identification."""

    def test_ceramide_classes_detected(self):
        """Test that ceramide-family lipids are detected."""
        from lipidetective.helpers.utils import is_lipid_class_with_slash

        assert is_lipid_class_with_slash("Cer 18:1") is True
        assert is_lipid_class_with_slash("HexCer 42:2") is True

    def test_sphingomyelin_detected(self):
        """Test that sphingomyelin is detected."""
        from lipidetective.helpers.utils import is_lipid_class_with_slash

        assert is_lipid_class_with_slash("SM 34:1") is True

    def test_phosphatidylcholine_not_detected(self):
        """Test that PC is not in the slash-class group."""
        from lipidetective.helpers.utils import is_lipid_class_with_slash

        assert is_lipid_class_with_slash("PC 34:1") is False


class TestSetSeeds:
    """Tests for reproducibility utilities."""

    def test_set_seeds_runs_without_error(self):
        """Test that set_seeds executes without raising exceptions."""
        from lipidetective.helpers.utils import set_seeds

        set_seeds(seed=123)

    def test_set_seeds_produces_reproducible_results(self):
        """Test that set_seeds produces reproducible random numbers."""
        import torch

        from lipidetective.helpers.utils import set_seeds

        set_seeds(seed=42)
        random_tensor_1 = torch.rand(5)

        set_seeds(seed=42)
        random_tensor_2 = torch.rand(5)

        assert torch.equal(random_tensor_1, random_tensor_2)


class TestIsMainProcess:
    """Tests for distributed processing detection."""

    def test_is_main_process_returns_bool(self):
        """Test is_main_process returns boolean."""
        from lipidetective.helpers.utils import is_main_process

        result = is_main_process()
        assert isinstance(result, bool)

    def test_is_main_process_true_without_env_vars(self, monkeypatch):
        """Should return True when LOCAL_RANK and NODE_RANK are not set."""
        from lipidetective.helpers.utils import is_main_process

        monkeypatch.delenv("LOCAL_RANK", raising=False)
        monkeypatch.delenv("NODE_RANK", raising=False)

        assert is_main_process() is True

    def test_is_main_process_false_with_local_rank(self, monkeypatch):
        """Should return False when LOCAL_RANK is set."""
        from lipidetective.helpers.utils import is_main_process

        monkeypatch.setenv("LOCAL_RANK", "1")
        monkeypatch.delenv("NODE_RANK", raising=False)

        assert is_main_process() is False


class TestWriteYaml:
    """Tests for YAML file writing."""

    def test_write_yaml_creates_file(self):
        """Test write_yaml creates a valid YAML file."""
        from lipidetective.helpers.utils import read_yaml, write_yaml

        data = {"model": "transformer", "epochs": 10}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            temp_path = f.name

        try:
            write_yaml(temp_path, data)
            loaded = read_yaml(temp_path)
            assert loaded["model"] == "transformer"
            assert loaded["epochs"] == 10
        finally:
            os.unlink(temp_path)

    def test_write_yaml_handles_nested_dict(self):
        """Test write_yaml handles nested dictionaries."""
        from lipidetective.helpers.utils import read_yaml, write_yaml

        data = {"model": {"type": "transformer", "params": {"d_model": 128}}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            temp_path = f.name

        try:
            write_yaml(temp_path, data)
            loaded = read_yaml(temp_path)
            assert loaded["model"]["type"] == "transformer"
            assert loaded["model"]["params"]["d_model"] == 128
        finally:
            os.unlink(temp_path)

    def test_write_yaml_handles_invalid_path(self, caplog):
        """Test write_yaml handles write errors gracefully."""
        import logging as _logging

        from lipidetective.helpers.utils import write_yaml

        # Try to write to an invalid path (directory that doesn't exist)
        invalid_path = "/nonexistent_dir_12345/config.yaml"
        data = {"key": "value"}

        # Should not raise, just log the exception
        with caplog.at_level(_logging.ERROR):
            write_yaml(invalid_path, data)

        assert "Failed to write YAML file" in caplog.text


class TestParseConfig:
    """Tests for parse_config function."""

    def test_parse_config_with_valid_args(self, monkeypatch, sample_yaml_file):
        """Test parse_config parses command line arguments."""
        import sys

        from lipidetective.helpers.utils import parse_config

        # Mock sys.argv
        monkeypatch.setattr(sys, "argv", ["lipidetective", "--config", sample_yaml_file])

        config, args = parse_config()

        assert config is not None
        assert config["model"] == "transformer"
        assert args.config == sample_yaml_file
        assert args.head_node_ip is None

    def test_parse_config_with_head_node_ip(self, monkeypatch, sample_yaml_file):
        """Test parse_config parses optional head_node_ip argument."""
        import sys

        from lipidetective.helpers.utils import parse_config

        monkeypatch.setattr(
            sys,
            "argv",
            ["lipidetective", "--config", sample_yaml_file, "--head_node_ip", "192.168.1.1"],
        )

        config, args = parse_config()

        assert args.head_node_ip == "192.168.1.1"


class TestResolveConfigPaths:
    """Tests for resolve_config_paths function."""

    def test_returns_copy_without_files_key(self):
        """Should return config copy when no files key exists."""
        from lipidetective.helpers.utils import resolve_config_paths

        config = {"model": "transformer", "epochs": 10}
        result = resolve_config_paths(config)

        assert result == config
        assert result is not config  # Should be a copy

    def test_resolves_data_paths(self):
        """Should resolve relative data paths."""
        from lipidetective.helpers.utils import resolve_config_paths

        config = {
            "files": {
                "train_input": "train.hdf5",
                "val_input": "val.hdf5",
            }
        }
        result = resolve_config_paths(config)

        assert os.path.isabs(result["files"]["train_input"])
        assert os.path.isabs(result["files"]["val_input"])

    def test_preserves_absolute_paths(self):
        """Should not modify absolute paths."""
        from lipidetective.helpers.utils import resolve_config_paths

        abs_path = "/absolute/path/to/data.hdf5"
        config = {"files": {"train_input": abs_path}}
        result = resolve_config_paths(config)

        assert result["files"]["train_input"] == abs_path

    def test_resolves_model_path(self):
        """Should resolve model paths."""
        from lipidetective.helpers.utils import resolve_config_paths

        config = {"files": {"saved_model": "model.pth"}}
        result = resolve_config_paths(config)

        assert os.path.isabs(result["files"]["saved_model"])

    def test_resolves_output_path(self):
        """Should resolve output paths."""
        from lipidetective.helpers.utils import resolve_config_paths

        config = {"files": {"output": "experiments/run1"}}
        result = resolve_config_paths(config)

        assert os.path.isabs(result["files"]["output"])

    def test_resolves_splitting_instructions_path(self):
        """Should resolve config paths for splitting instructions."""
        from lipidetective.helpers.utils import resolve_config_paths

        config = {"files": {"splitting_instructions": "splits/train_val.yaml"}}
        result = resolve_config_paths(config)

        assert os.path.isabs(result["files"]["splitting_instructions"])

    def test_handles_empty_paths(self):
        """Should handle empty or None paths gracefully."""
        from lipidetective.helpers.utils import resolve_config_paths

        config = {"files": {"train_input": "", "val_input": None}}
        result = resolve_config_paths(config)

        assert result["files"]["train_input"] == ""
        assert result["files"]["val_input"] is None


class TestLipidClassDetectionExtended:
    """Extended tests for lipid class identification."""

    def test_all_ceramide_variants(self):
        """Test all ceramide-related classes are detected."""
        from lipidetective.helpers.utils import is_lipid_class_with_slash

        ceramide_classes = [
            "Cer",
            "HexCer",
            "LacCer",
            "GalCer",
            "SHexCer",
            "GlcCer",
            "Hex2Cer",
            "Hex3Cer",
        ]
        for cls in ceramide_classes:
            assert is_lipid_class_with_slash(f"{cls} 18:1/16:0") is True

    def test_ganglioside_classes(self):
        """Test ganglioside classes are detected."""
        from lipidetective.helpers.utils import is_lipid_class_with_slash

        gangliosides = ["GM1", "GM3", "GD1a", "GD1b", "GT1b"]
        for cls in gangliosides:
            assert is_lipid_class_with_slash(f"{cls} 34:1") is True

    def test_other_slash_classes(self):
        """Test other lipid classes that use slash notation."""
        from lipidetective.helpers.utils import is_lipid_class_with_slash

        other_classes = ["EPC", "IPC", "SE", "FAHFA"]
        for cls in other_classes:
            assert is_lipid_class_with_slash(f"{cls} 34:1") is True

    def test_glycerophospholipids_not_detected(self):
        """Test that glycerophospholipids are not in slash group."""
        from lipidetective.helpers.utils import is_lipid_class_with_slash

        gp_classes = ["PC", "PE", "PS", "PI", "PG", "PA", "LPC", "LPE"]
        for cls in gp_classes:
            assert is_lipid_class_with_slash(f"{cls} 34:1") is False

    def test_triglycerides_not_detected(self):
        """Test that triglycerides are not in slash group."""
        from lipidetective.helpers.utils import is_lipid_class_with_slash

        assert is_lipid_class_with_slash("TG 16:0_18:1_18:2") is False
        assert is_lipid_class_with_slash("DG 16:0_18:1") is False


class TestModelMetadata:
    """Tests for model metadata extract/validate functions."""

    def _transformer_config(self):
        return {
            "model": "transformer",
            "input_embedding": {"n_peaks": 30, "max_mz": 1600, "decimal_accuracy": 1},
            "transformer": {
                "d_model": 32,
                "num_heads": 4,
                "dropout": 0.1,
                "ffn_hidden": 64,
                "num_layers": 2,
                "output_seq_length": 11,
            },
            "training": {"batch": 4, "epochs": 2},
        }

    def _convolutional_config(self):
        return {
            "model": "convolutional",
            "input_embedding": {"type": "peaks", "n_peaks": 100, "max_mz": 1600},
            "convolutional": {
                "channels_conv_1": 16,
                "kernel_size_1": [2, 3],
                "lin_layer_1": 64,
            },
            "training": {"batch": 4, "epochs": 2},
        }

    def test_extract_transformer_metadata(self):
        """Correct keys extracted for transformer config."""
        from lipidetective.helpers.utils import extract_model_metadata

        config = self._transformer_config()
        metadata = extract_model_metadata(config)

        assert metadata["model"] == "transformer"
        assert "input_embedding" in metadata
        assert "transformer" in metadata
        assert metadata["transformer"]["d_model"] == 32
        # Training params should NOT be in metadata
        assert "training" not in metadata

    def test_extract_convolutional_metadata(self):
        """Correct keys extracted for convolutional config."""
        from lipidetective.helpers.utils import extract_model_metadata

        config = self._convolutional_config()
        metadata = extract_model_metadata(config)

        assert metadata["model"] == "convolutional"
        assert "input_embedding" in metadata
        assert "convolutional" in metadata
        assert "transformer" not in metadata

    def test_extract_includes_version(self):
        """Metadata contains lipidetective_version."""
        from lipidetective.helpers.utils import extract_model_metadata

        config = self._transformer_config()
        metadata = extract_model_metadata(config)

        assert "lipidetective_version" in metadata
        assert isinstance(metadata["lipidetective_version"], str)

    def test_validate_matching_config(self, tmp_path, caplog):
        """No warnings logged when config matches saved metadata."""
        from lipidetective.helpers.utils import (
            extract_model_metadata,
            validate_model_metadata,
            write_yaml,
        )

        config = self._transformer_config()
        metadata = extract_model_metadata(config)
        sidecar = str(tmp_path / "model_config.yaml")
        write_yaml(sidecar, metadata)

        with caplog.at_level(_logging.WARNING):
            result = validate_model_metadata(config.copy(), sidecar)

        assert result["transformer"]["d_model"] == 32
        assert "override" not in caplog.text.lower()

    def test_validate_mismatched_config_overrides(self, tmp_path, caplog):
        """Differing keys get overridden with warnings."""
        from lipidetective.helpers.utils import (
            extract_model_metadata,
            validate_model_metadata,
            write_yaml,
        )

        config = self._transformer_config()
        metadata = extract_model_metadata(config)
        sidecar = str(tmp_path / "model_config.yaml")
        write_yaml(sidecar, metadata)

        # Change config to have different values
        modified = self._transformer_config()
        modified["transformer"]["d_model"] = 64
        modified["input_embedding"]["n_peaks"] = 50

        with caplog.at_level(_logging.WARNING):
            result = validate_model_metadata(modified, sidecar)

        # Should be overridden to saved values
        assert result["transformer"]["d_model"] == 32
        assert result["input_embedding"]["n_peaks"] == 30
        assert "override" in caplog.text.lower()

    def test_validate_missing_sidecar(self, caplog):
        """Returns config unchanged when sidecar doesn't exist."""
        from lipidetective.helpers.utils import validate_model_metadata

        config = self._transformer_config()
        original_d_model = config["transformer"]["d_model"]

        with caplog.at_level(_logging.DEBUG):
            result = validate_model_metadata(config, "/nonexistent/model_config.yaml")

        assert result["transformer"]["d_model"] == original_d_model
        assert result is config  # same object, unchanged


class TestSetupLogging:
    """Tests for setup_logging function."""

    @pytest.fixture(autouse=True)
    def _reset_root_logger(self):
        """Reset root logger handlers after each test."""
        yield
        root = _logging.getLogger()
        for handler in root.handlers[:]:
            root.removeHandler(handler)
        root.setLevel(_logging.WARNING)

    def test_default_config_sets_info_level(self):
        """Default config (no logging key) should set INFO level with 1 handler."""
        from lipidetective.helpers.utils import setup_logging

        setup_logging({"model": "transformer"})

        root = _logging.getLogger()
        assert root.level == _logging.INFO
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0], _logging.StreamHandler)

    def test_warning_level(self):
        """Setting level to WARNING should configure root logger accordingly."""
        from lipidetective.helpers.utils import setup_logging

        setup_logging({"logging": {"level": "WARNING"}})

        root = _logging.getLogger()
        assert root.level == _logging.WARNING

    def test_debug_level(self):
        """Setting level to DEBUG should configure root logger accordingly."""
        from lipidetective.helpers.utils import setup_logging

        setup_logging({"logging": {"level": "DEBUG"}})

        root = _logging.getLogger()
        assert root.level == _logging.DEBUG

    def test_file_handler(self, tmp_path):
        """Setting file should add a FileHandler that writes output."""
        from lipidetective.helpers.utils import setup_logging

        log_file = str(tmp_path / "test.log")
        setup_logging({"logging": {"file": log_file}})

        _logging.info("test message")

        root = _logging.getLogger()
        assert len(root.handlers) == 2

        # Flush and check file contents
        for handler in root.handlers:
            handler.flush()

        with open(log_file) as f:
            contents = f.read()
        assert "test message" in contents

    def test_logging_null_falls_back_to_info(self):
        """``logging: null`` in YAML should not crash, falls back to INFO."""
        from lipidetective.helpers.utils import setup_logging

        setup_logging({"logging": None})

        root = _logging.getLogger()
        assert root.level == _logging.INFO
        assert len(root.handlers) == 1

    def test_invalid_level_falls_back_to_info(self):
        """An invalid level name should fall back to INFO."""
        from lipidetective.helpers.utils import setup_logging

        setup_logging({"logging": {"level": "BANANA"}})

        root = _logging.getLogger()
        assert root.level == _logging.INFO
