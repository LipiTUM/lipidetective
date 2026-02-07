"""Tests for PredictionDataset class."""

import json

import numpy as np
import pytest
import torch

from lipidetective.workflow.prediction_dataset import PredictionDataset


@pytest.fixture
def prediction_config():
    """Config for prediction dataset tests."""
    return {
        "input_embedding": {
            "n_peaks": 50,
            "decimal_accuracy": 1,
        }
    }


@pytest.fixture
def sample_json_spectra(tmp_path):
    """Create a temporary JSON file with sample spectra."""
    spectra = [
        {
            "index": 0,
            "precursor": 760.5,
            "mz": [100.1, 200.2, 300.3, 400.4, 500.5],
            "intensity": [1000.0, 2000.0, 1500.0, 500.0, 3000.0],
            "polarity": "+",
        },
        {
            "index": 1,
            "precursor": 820.6,
            "mz": [150.1, 250.2, 350.3],
            "intensity": [800.0, 1200.0, 600.0],
            "polarity": "-",
        },
    ]
    json_path = tmp_path / "test_spectra.json"
    with open(json_path, "w") as f:
        json.dump(spectra, f)
    return str(json_path)


class TestPredictionDatasetInitialization:
    """Tests for PredictionDataset initialization."""

    def test_initialization_with_json(self, sample_json_spectra, prediction_config):
        """Should initialize from JSON file."""
        dataset = PredictionDataset(sample_json_spectra, prediction_config)

        assert dataset.file_name == "test_spectra.json"
        assert dataset.dataset_len == 2
        assert dataset.file is not None

    def test_initialization_stores_config(self, sample_json_spectra, prediction_config):
        """Should store config for later use."""
        dataset = PredictionDataset(sample_json_spectra, prediction_config)

        assert dataset.config == prediction_config


class TestPredictionDatasetLen:
    """Tests for __len__ method."""

    def test_len_returns_correct_count(self, sample_json_spectra, prediction_config):
        """__len__ should return number of spectra."""
        dataset = PredictionDataset(sample_json_spectra, prediction_config)

        assert len(dataset) == 2


class TestPredictionDatasetGetItem:
    """Tests for __getitem__ method."""

    def test_getitem_returns_dict(self, sample_json_spectra, prediction_config):
        """__getitem__ should return dict with features and info."""
        dataset = PredictionDataset(sample_json_spectra, prediction_config)

        sample = dataset[0]

        assert "features" in sample
        assert "info" in sample

    def test_getitem_features_is_tensor(self, sample_json_spectra, prediction_config):
        """Features should be a torch tensor."""
        dataset = PredictionDataset(sample_json_spectra, prediction_config)

        sample = dataset[0]

        assert isinstance(sample["features"], torch.Tensor)

    def test_getitem_info_contains_metadata(self, sample_json_spectra, prediction_config):
        """Info should contain spectrum metadata."""
        dataset = PredictionDataset(sample_json_spectra, prediction_config)

        sample = dataset[0]
        info = sample["info"]

        assert info["index"] == 0
        assert info["file"] == "test_spectra.json"
        assert info["polarity"] == "+"
        assert "precursor" in info

    def test_getitem_second_spectrum(self, sample_json_spectra, prediction_config):
        """Should correctly retrieve second spectrum."""
        dataset = PredictionDataset(sample_json_spectra, prediction_config)

        sample = dataset[1]
        info = sample["info"]

        assert info["index"] == 1
        assert info["polarity"] == "-"


class TestPredictionDatasetProcessInput:
    """Tests for process_input method."""

    def test_process_input_json(self, sample_json_spectra, prediction_config):
        """Should process JSON files."""
        dataset = PredictionDataset(sample_json_spectra, prediction_config)

        # File was processed during init
        assert dataset.file is not None
        assert len(dataset.file) == 2

    def test_process_input_unsupported_format_raises(self, tmp_path, prediction_config):
        """Unsupported formats raise ValueError with helpful message."""
        unsupported_file = tmp_path / "test.txt"
        unsupported_file.write_text("some content")

        with pytest.raises(ValueError, match="Unsupported format"):
            PredictionDataset(str(unsupported_file), prediction_config)


class TestPredictionDatasetProcessJson:
    """Tests for process_json method."""

    def test_process_json_loads_spectra(self, sample_json_spectra, prediction_config):
        """Should load spectra from JSON file."""
        dataset = PredictionDataset(sample_json_spectra, prediction_config)

        assert len(dataset.file) == 2
        assert dataset.file[0]["precursor"] == 760.5
        assert dataset.file[1]["precursor"] == 820.6


class TestPredictionDatasetGetNHighestPeaks:
    """Tests for get_n_highest_peaks method."""

    def test_returns_tensor_of_correct_length(self, sample_json_spectra, prediction_config):
        """Should return tensor with n_peaks elements."""
        dataset = PredictionDataset(sample_json_spectra, prediction_config)

        mz_array = np.array([100.1, 200.2, 300.3, 400.4, 500.5])
        intensity_array = np.array([1000.0, 2000.0, 1500.0, 500.0, 3000.0])
        precursor = 760.5

        features = dataset.get_n_highest_peaks(mz_array, intensity_array, precursor)

        assert len(features) == prediction_config["input_embedding"]["n_peaks"]

    def test_selects_highest_intensity_peaks(self, sample_json_spectra, prediction_config):
        """Should select peaks with highest intensity."""
        # Use smaller n_peaks for this test
        config = {"input_embedding": {"n_peaks": 3, "decimal_accuracy": 1}}
        dataset = PredictionDataset(sample_json_spectra, config)

        mz_array = np.array([100.1, 200.2, 300.3, 400.4, 500.5])
        intensity_array = np.array([1000.0, 2000.0, 1500.0, 500.0, 3000.0])
        precursor = 760.5

        features = dataset.get_n_highest_peaks(mz_array, intensity_array, precursor)

        # Highest intensity peaks are at 500.5, 200.2, 300.3
        # With decimal_accuracy=1: 5005, 2002, 3003
        assert 5005 in features.tolist()
        assert 2002 in features.tolist()

    def test_pads_when_fewer_peaks_than_n(self, sample_json_spectra, prediction_config):
        """Should pad with zeros when fewer peaks than n_peaks."""
        config = {"input_embedding": {"n_peaks": 10, "decimal_accuracy": 1}}
        dataset = PredictionDataset(sample_json_spectra, config)

        mz_array = np.array([100.1, 200.2])
        intensity_array = np.array([1000.0, 2000.0])
        precursor = 760.5

        features = dataset.get_n_highest_peaks(mz_array, intensity_array, precursor)

        assert len(features) == 10
        # Most values should be zero (padding)
        zero_count = (features == 0).sum().item()
        assert zero_count >= 7  # At least 7 zeros (10 - 2 peaks - possibly precursor)

    def test_includes_precursor_if_not_present(self, sample_json_spectra, prediction_config):
        """Should add precursor m/z if not in peaks."""
        config = {"input_embedding": {"n_peaks": 3, "decimal_accuracy": 1}}
        dataset = PredictionDataset(sample_json_spectra, config)

        mz_array = np.array([100.1, 200.2, 300.3])
        intensity_array = np.array([1000.0, 2000.0, 1500.0])
        precursor = 760.5  # Not in mz_array

        features = dataset.get_n_highest_peaks(mz_array, intensity_array, precursor)

        # Precursor (7605) should be added to last position
        assert 7605 in features.tolist()

    def test_decimal_accuracy_scaling(self, sample_json_spectra, prediction_config):
        """Should scale m/z values by decimal accuracy."""
        config = {"input_embedding": {"n_peaks": 3, "decimal_accuracy": 2}}
        dataset = PredictionDataset(sample_json_spectra, config)

        mz_array = np.array([100.12, 200.25])
        intensity_array = np.array([1000.0, 2000.0])
        precursor = 760.55

        features = dataset.get_n_highest_peaks(mz_array, intensity_array, precursor)

        # With decimal_accuracy=2: 10012, 20025, 76055
        assert 20025 in features.tolist()  # Highest intensity


class TestPredictionDatasetIteration:
    """Tests for iterating over dataset."""

    def test_can_iterate_over_dataset(self, sample_json_spectra, prediction_config):
        """Should be able to iterate using DataLoader-like pattern."""
        dataset = PredictionDataset(sample_json_spectra, prediction_config)

        samples = [dataset[i] for i in range(len(dataset))]

        assert len(samples) == 2
        assert all("features" in s for s in samples)
        assert all("info" in s for s in samples)
