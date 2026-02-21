"""Tests for PrecomputedDataset class."""

import torch

from lipidetective.workflow.h5_dataset import PrecomputedDataset


class TestPrecomputedDatasetInitialization:
    """Tests for PrecomputedDataset initialization."""

    def test_initialization_loads_into_memory(
        self, transformer_config, lipid_library, test_parquet_file
    ):
        """Dataset should load all matching entries into memory at init time."""
        parquet_path, dataset_names, _ = test_parquet_file

        dataset = PrecomputedDataset(
            config=transformer_config,
            dataset_names=dataset_names,
            lipid_librarian=lipid_library,
            file_path=parquet_path,
        )

        assert len(dataset.features_cache) == len(dataset_names)
        assert len(dataset.attrs_cache) == len(dataset_names)

    def test_initialization_filters_by_dataset_names(
        self, transformer_config, lipid_library, test_parquet_file
    ):
        """Only the requested dataset names should be cached, not the full file."""
        parquet_path, dataset_names, _ = test_parquet_file
        subset = dataset_names[:2]

        dataset = PrecomputedDataset(
            config=transformer_config,
            dataset_names=subset,
            lipid_librarian=lipid_library,
            file_path=parquet_path,
        )

        assert len(dataset.features_cache) == 2
        assert set(dataset.features_cache.keys()) == set(subset)
        assert dataset_names[2] not in dataset.features_cache

    def test_length_matches_dataset_names(
        self, transformer_config, lipid_library, test_parquet_file
    ):
        """Length should match the number of requested dataset names."""
        parquet_path, dataset_names, _ = test_parquet_file

        dataset = PrecomputedDataset(
            config=transformer_config,
            dataset_names=dataset_names,
            lipid_librarian=lipid_library,
            file_path=parquet_path,
        )

        assert len(dataset) == len(dataset_names)
        assert len(dataset) == 3


class TestPrecomputedDatasetGetItem:
    """Tests for PrecomputedDataset __getitem__."""

    def test_getitem_returns_required_keys(
        self, transformer_config, lipid_library, test_parquet_file
    ):
        """Getitem should return the same keys as H5Dataset."""
        parquet_path, dataset_names, _ = test_parquet_file

        dataset = PrecomputedDataset(
            config=transformer_config,
            dataset_names=dataset_names,
            lipid_librarian=lipid_library,
            file_path=parquet_path,
        )

        sample = dataset[0]

        assert "features" in sample
        assert "label" in sample
        assert "info" in sample
        assert "dataset_path" in sample

    def test_getitem_features_shape_matches_h5dataset(
        self, transformer_config, lipid_library, test_parquet_file
    ):
        """Features should be a 1D IntTensor of n_peaks, matching H5Dataset output."""
        parquet_path, dataset_names, _ = test_parquet_file

        dataset = PrecomputedDataset(
            config=transformer_config,
            dataset_names=dataset_names,
            lipid_librarian=lipid_library,
            file_path=parquet_path,
        )

        sample = dataset[0]
        n_peaks = transformer_config["input_embedding"]["n_peaks"]

        assert sample["features"].shape == (n_peaks,)
        assert sample["features"].dtype == torch.int32

    def test_getitem_label_shape(self, transformer_config, lipid_library, test_parquet_file):
        """Label should be padded to output_seq_length."""
        parquet_path, dataset_names, _ = test_parquet_file

        dataset = PrecomputedDataset(
            config=transformer_config,
            dataset_names=dataset_names,
            lipid_librarian=lipid_library,
            file_path=parquet_path,
        )

        sample = dataset[0]
        seq_length = transformer_config["transformer"]["output_seq_length"]

        assert sample["label"].shape == (seq_length,)

    def test_getitem_dataset_path_is_index(
        self, transformer_config, lipid_library, test_parquet_file
    ):
        """dataset_path should encode the index of the sample."""
        parquet_path, dataset_names, _ = test_parquet_file

        dataset = PrecomputedDataset(
            config=transformer_config,
            dataset_names=dataset_names,
            lipid_librarian=lipid_library,
            file_path=parquet_path,
        )

        assert dataset[0]["dataset_path"].item() == 0
        assert dataset[1]["dataset_path"].item() == 1
        assert dataset[2]["dataset_path"].item() == 2

    def test_getitem_info_contains_dataset_name(
        self, transformer_config, lipid_library, test_parquet_file
    ):
        """info dict should contain the dataset name for traceability."""
        parquet_path, dataset_names, _ = test_parquet_file

        dataset = PrecomputedDataset(
            config=transformer_config,
            dataset_names=dataset_names,
            lipid_librarian=lipid_library,
            file_path=parquet_path,
        )

        sample = dataset[0]
        assert sample["info"]["dataset_name"] == dataset_names[0]


class TestPrecomputedDatasetCache:
    """Tests for features and attributes caching."""

    def test_features_are_cached_by_name(
        self, transformer_config, lipid_library, test_parquet_file
    ):
        """Each dataset name should map to a list of integer features in the cache."""
        parquet_path, dataset_names, _ = test_parquet_file

        dataset = PrecomputedDataset(
            config=transformer_config,
            dataset_names=dataset_names,
            lipid_librarian=lipid_library,
            file_path=parquet_path,
        )

        for name in dataset_names:
            assert name in dataset.features_cache
            assert isinstance(dataset.features_cache[name], list)
            assert all(isinstance(v, int) for v in dataset.features_cache[name])

    def test_attrs_cache_contains_lipid_and_adduct(
        self, transformer_config, lipid_library, test_parquet_file
    ):
        """Attributes cache should store lipid_species and adduct for each sample."""
        parquet_path, dataset_names, sample_lipids = test_parquet_file

        dataset = PrecomputedDataset(
            config=transformer_config,
            dataset_names=dataset_names,
            lipid_librarian=lipid_library,
            file_path=parquet_path,
        )

        for i, name in enumerate(dataset_names):
            attrs = dataset.attrs_cache[name]
            assert attrs["lipid_species"] == sample_lipids[i][0]
            assert attrs["adduct"] == sample_lipids[i][1]
