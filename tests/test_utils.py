"""Unit tests for helpers/utils.py functions."""

import numpy as np


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
