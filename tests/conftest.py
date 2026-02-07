"""Pytest configuration and shared fixtures."""

import os
import tempfile

import h5py
import numpy as np
import pytest
from lipidetective.helpers.lipid_library import LipidLibrary


@pytest.fixture
def sample_yaml_file():
    """Create a temporary YAML file for testing."""
    content = """
model: transformer
workflow:
  train: true
  validate: false
  test: false
  predict: false
  tune: false
transformer:
  d_model: 128
  num_heads: 4
  num_layers: 2
  output_seq_length: 20
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(content)
        f.flush()
        yield f.name
    os.unlink(f.name)


@pytest.fixture(scope="session")
def lipid_library():
    """Shared LipidLibrary instance using real YAML files."""
    return LipidLibrary()


@pytest.fixture
def transformer_config():
    """Minimal config for transformer model tests."""
    return {
        "model": "transformer",
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


@pytest.fixture
def regression_config():
    """Minimal config for CNN/FFN regression model tests."""
    return {
        "model": "convolutional",
        "input_embedding": {
            "type": "peaks",
            "n_peaks": 100,
            "max_mz": 1600,
            "decimal_accuracy": 1,
        },
        "convolutional": {
            "channels_conv_1": 16,
            "channels_conv_2": 32,
            "channels_conv_3": 64,
            "kernel_size_1": [2, 3],
            "kernel_size_2": [1, 3],
            "kernel_size_3": [1, 3],
            "stride_1": 1,
            "stride_2": 1,
            "stride_3": 1,
            "lin_layer_1": 64,
            "lin_layer_2": 32,
        },
        "feedforward": {
            "layer_1_size": 64,
            "layer_2_size": 32,
            "layer_3_size": 16,
        },
    }


@pytest.fixture
def test_hdf5_file(tmp_path, lipid_library):
    """Create a temporary HDF5 file with 3 sample spectra for testing."""
    hdf5_path = tmp_path / "test_dataset.hdf5"

    # Get a valid lipid species from the library for realistic test data
    sample_lipids = [
        ("PC 16:0_18:1", "[M+H]+", "pos"),
        ("PE 18:0_18:2", "[M+H]+", "pos"),
        ("SM d18:1/16:0", "[M+H]+", "pos"),
    ]

    with h5py.File(hdf5_path, "w") as f:
        all_datasets = f.create_group("all_datasets")

        for i, (lipid_species, adduct, polarity) in enumerate(sample_lipids):
            # Create synthetic spectrum: m/z values from 100-800, random intensities
            n_peaks = 50
            mz_values = np.sort(np.random.uniform(100, 800, n_peaks))
            intensities = np.random.uniform(0, 1000, n_peaks)
            spectrum = np.array([mz_values, intensities])

            dataset = all_datasets.create_dataset(f"spectrum_{i}", data=spectrum)
            dataset.attrs["lipid_species"] = lipid_species
            dataset.attrs["adduct"] = adduct
            dataset.attrs["polarity"] = polarity
            dataset.attrs["precursor"] = float(np.random.uniform(700, 900))

    dataset_names = [f"spectrum_{i}" for i in range(3)]

    return str(hdf5_path), dataset_names, sample_lipids


@pytest.fixture
def rf_instance(monkeypatch):
    """Create a RandomForest instance with mocked YAML loading."""

    from lipidetective.models.random_forest import RandomForest

    # Mock read_yaml to return test data
    mock_lipid_species = {
        "PC 34:1": {"headgroup": "PC", "fatty_acid_sn1": "16:0", "fatty_acid_sn2": "18:1"},
        "PE 36:2": {"headgroup": "PE", "fatty_acid_sn1": "18:1", "fatty_acid_sn2": "18:1"},
    }
    mock_headgroups = {"PC": 184.07, "PE": 141.02}
    mock_fatty_acids = {
        "16:0": {"mono_mass": 256.24},
        "18:1": {"mono_mass": 282.26},
    }

    def mock_read_yaml(path):
        if "molecular_lipid_species" in path:
            return mock_lipid_species
        elif "headgroups" in path:
            return mock_headgroups
        elif "sidechains" in path:
            return mock_fatty_acids
        return {}

    monkeypatch.setattr("lipidetective.models.random_forest.read_yaml", mock_read_yaml)

    config = {
        "files": {"output": "/tmp", "train_input": None, "splitting_instructions": None},
        "random_forest": {"type": "single_classifier"},
    }

    return RandomForest(config)


@pytest.fixture
def lightning_config():
    """Config for LightningModule tests."""
    return {
        "model": "transformer",
        "workflow": {"load_model": False, "tune": False},
        "training": {
            "batch": 4,
            "epochs": 2,
            "learning_rate": 0.001,
            "lr_step": 5,
        },
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
