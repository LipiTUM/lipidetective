"""Tests for H5Dataset class."""

import numpy as np
import torch

from lipidetective.workflow.h5_dataset import H5Dataset


class TestH5DatasetInitialization:
    """Tests for H5Dataset initialization."""

    def test_initialization_sets_attributes(
        self, transformer_config, lipid_library, test_hdf5_file
    ):
        """Dataset should properly initialize all attributes."""
        hdf5_path, dataset_names, _ = test_hdf5_file

        dataset = H5Dataset(
            config=transformer_config,
            dataset_names=dataset_names,
            lipid_librarian=lipid_library,
            file_path=hdf5_path,
        )

        assert dataset.file_path == hdf5_path
        assert dataset.dataset_names == dataset_names
        assert dataset.network_type == "transformer"
        assert dataset.hdf5_file is None  # Lazy loaded


class TestH5DatasetLength:
    """Tests for H5Dataset __len__."""

    def test_length_returns_dataset_count(self, transformer_config, lipid_library, test_hdf5_file):
        """Length should match number of dataset names."""
        hdf5_path, dataset_names, _ = test_hdf5_file

        dataset = H5Dataset(
            config=transformer_config,
            dataset_names=dataset_names,
            lipid_librarian=lipid_library,
            file_path=hdf5_path,
        )

        assert len(dataset) == len(dataset_names)
        assert len(dataset) == 3


class TestH5DatasetGetItem:
    """Tests for H5Dataset __getitem__."""

    def test_getitem_returns_dict_with_required_keys(
        self, transformer_config, lipid_library, test_hdf5_file
    ):
        """Getitem should return dict with features, label, info, dataset_path."""
        hdf5_path, dataset_names, _ = test_hdf5_file

        dataset = H5Dataset(
            config=transformer_config,
            dataset_names=dataset_names,
            lipid_librarian=lipid_library,
            file_path=hdf5_path,
        )

        sample = dataset[0]

        assert "features" in sample
        assert "label" in sample
        assert "info" in sample
        assert "dataset_path" in sample

    def test_getitem_features_shape_transformer(
        self, transformer_config, lipid_library, test_hdf5_file
    ):
        """Transformer features should be 1D tensor of n_peaks integers."""
        hdf5_path, dataset_names, _ = test_hdf5_file

        dataset = H5Dataset(
            config=transformer_config,
            dataset_names=dataset_names,
            lipid_librarian=lipid_library,
            file_path=hdf5_path,
        )

        sample = dataset[0]
        n_peaks = transformer_config["input_embedding"]["n_peaks"]

        assert sample["features"].shape == (n_peaks,)
        assert sample["features"].dtype == torch.int64

    def test_getitem_label_shape_transformer(
        self, transformer_config, lipid_library, test_hdf5_file
    ):
        """Transformer label should be padded to output_seq_length."""
        hdf5_path, dataset_names, _ = test_hdf5_file

        dataset = H5Dataset(
            config=transformer_config,
            dataset_names=dataset_names,
            lipid_librarian=lipid_library,
            file_path=hdf5_path,
        )

        sample = dataset[0]
        seq_length = transformer_config["transformer"]["output_seq_length"]

        assert sample["label"].shape == (seq_length,)

    def test_getitem_regression_features_shape(
        self, regression_config, lipid_library, test_hdf5_file
    ):
        """Regression features should include m/z, intensity, polarity, precursor."""
        hdf5_path, dataset_names, sample_lipids = test_hdf5_file

        # Regression config needs valid lipid in molecular_lipid_species
        # Use a lipid we know exists
        sample_lipid = list(lipid_library.molecular_lipid_species.keys())[0]

        # Re-create HDF5 with a valid molecular lipid species
        import h5py

        with h5py.File(hdf5_path, "r+") as f:
            for name in dataset_names:
                f[f"/all_datasets/{name}"].attrs["lipid_species"] = sample_lipid

        dataset = H5Dataset(
            config=regression_config,
            dataset_names=dataset_names,
            lipid_librarian=lipid_library,
            file_path=hdf5_path,
        )

        sample = dataset[0]
        n_peaks = regression_config["input_embedding"]["n_peaks"]

        # Features should be (2, n_peaks+1) transposed to (n_peaks+1, 2)
        # Actually the code does features.T so it's (2, n_peaks+1)
        assert sample["features"].shape[1] == n_peaks + 1
        assert sample["features"].dtype == torch.float32

    def test_getitem_opens_hdf5_lazily(self, transformer_config, lipid_library, test_hdf5_file):
        """HDF5 file should only open on first access."""
        hdf5_path, dataset_names, _ = test_hdf5_file

        dataset = H5Dataset(
            config=transformer_config,
            dataset_names=dataset_names,
            lipid_librarian=lipid_library,
            file_path=hdf5_path,
        )

        assert dataset.hdf5_file is None
        _ = dataset[0]
        assert dataset.hdf5_file is not None


class TestGetNHighestPeaks:
    """Tests for get_n_highest_peaks method."""

    def test_returns_n_peaks(self, transformer_config, lipid_library, test_hdf5_file):
        """Should return exactly n_peaks peaks."""
        hdf5_path, dataset_names, _ = test_hdf5_file

        dataset = H5Dataset(
            config=transformer_config,
            dataset_names=dataset_names,
            lipid_librarian=lipid_library,
            file_path=hdf5_path,
        )

        # Create a test spectrum
        mz = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        intensity = np.array([10.0, 50.0, 30.0, 20.0, 40.0])
        spectrum = np.array([mz, intensity])

        n_peaks = 3
        result = dataset.get_n_highest_peaks(spectrum, n_peaks)

        assert result.shape[0] == n_peaks

    def test_pads_when_insufficient_peaks(self, transformer_config, lipid_library, test_hdf5_file):
        """Should pad with zeros when spectrum has fewer peaks than n_peaks."""
        hdf5_path, dataset_names, _ = test_hdf5_file

        dataset = H5Dataset(
            config=transformer_config,
            dataset_names=dataset_names,
            lipid_librarian=lipid_library,
            file_path=hdf5_path,
        )

        mz = np.array([100.0, 200.0])
        intensity = np.array([10.0, 50.0])
        spectrum = np.array([mz, intensity])

        n_peaks = 5
        result = dataset.get_n_highest_peaks(spectrum, n_peaks)

        assert result.shape[0] == n_peaks
        # Last entries should be zero (padding)
        assert result[-1, 0] == 0.0
        assert result[-1, 1] == 0.0


class TestBinSpectrum:
    """Tests for bin_spectrum method."""

    def test_bins_by_decimal_accuracy(self, transformer_config, lipid_library, test_hdf5_file):
        """Should truncate m/z values to decimal_accuracy and sum intensities."""
        hdf5_path, dataset_names, _ = test_hdf5_file

        dataset = H5Dataset(
            config=transformer_config,
            dataset_names=dataset_names,
            lipid_librarian=lipid_library,
            file_path=hdf5_path,
        )

        # With decimal_accuracy=1, 100.11 and 100.19 should bin to 100.1
        mz = np.array([100.11, 100.19, 200.5])
        intensity = np.array([10.0, 20.0, 30.0])
        spectrum = np.array([mz, intensity])

        result = dataset.bin_spectrum(spectrum)

        # Should have 2 unique bins: 100.1 (combined) and 200.5
        assert len(result) == 2

    def test_sorted_by_intensity_descending(
        self, transformer_config, lipid_library, test_hdf5_file
    ):
        """Binned spectrum should be sorted by intensity (highest first)."""
        hdf5_path, dataset_names, _ = test_hdf5_file

        dataset = H5Dataset(
            config=transformer_config,
            dataset_names=dataset_names,
            lipid_librarian=lipid_library,
            file_path=hdf5_path,
        )

        mz = np.array([100.0, 200.0, 300.0])
        intensity = np.array([10.0, 50.0, 30.0])
        spectrum = np.array([mz, intensity])

        result = dataset.bin_spectrum(spectrum)

        # First peak should have highest intensity (50.0)
        assert result[0, 1] == 50.0
        # Intensities should be descending
        for i in range(len(result) - 1):
            assert result[i, 1] >= result[i + 1, 1]
