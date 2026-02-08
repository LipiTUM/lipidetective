"""Smoke tests to verify the package loads correctly."""


class TestPackageImports:
    """Test that core package modules can be imported."""

    def test_import_main_package(self):
        """Verify the main lipidetective package is importable."""
        import lipidetective

        assert lipidetective is not None

    def test_import_helpers_utils(self):
        """Verify helpers.utils module is importable."""
        from lipidetective.helpers import utils

        assert utils is not None

    def test_import_models(self):
        """Verify model modules are importable."""
        from lipidetective.models import (
            convolutional_network,
            feedforward_network,
            transformer_network,
        )

        assert transformer_network is not None
        assert convolutional_network is not None
        assert feedforward_network is not None

    def test_import_workflow(self):
        """Verify workflow modules are importable."""
        from lipidetective.workflow import h5_dataset, lightning_module, trainer

        assert trainer is not None
        assert lightning_module is not None
        assert h5_dataset is not None


class TestPublicAPIExports:
    """Test that public API exports from __init__.py work correctly."""

    def test_version_metadata_accessible(self):
        """Verify version metadata is accessible from top-level package."""
        import lipidetective

        assert hasattr(lipidetective, "__version__")
        assert hasattr(lipidetective, "__author__")
        assert hasattr(lipidetective, "__license__")
        assert hasattr(lipidetective, "__url__")
        assert isinstance(lipidetective.__version__, str)
        assert len(lipidetective.__version__) > 0

    def test_model_classes_importable_from_top_level(self):
        """Verify model classes can be imported from top-level package."""
        from lipidetective import (
            ConvolutionalNetwork,
            FeedForwardNetwork,
            RandomForest,
            TransformerNetwork,
        )

        assert TransformerNetwork is not None
        assert ConvolutionalNetwork is not None
        assert FeedForwardNetwork is not None
        assert RandomForest is not None

    def test_workflow_classes_importable_from_top_level(self):
        """Verify workflow classes can be imported from top-level package."""
        from lipidetective import (
            H5Dataset,
            LightningModule,
            PredictionDataset,
            Trainer,
        )

        assert Trainer is not None
        assert H5Dataset is not None
        assert PredictionDataset is not None
        assert LightningModule is not None

    def test_helper_utilities_importable_from_top_level(self):
        """Verify helper utilities can be imported from top-level package."""
        from lipidetective import (
            LipidLibrary,
            read_yaml,
            resolve_config_paths,
            set_seeds,
            write_yaml,
        )

        assert LipidLibrary is not None
        assert callable(read_yaml)
        assert callable(write_yaml)
        assert callable(resolve_config_paths)
        assert callable(set_seeds)

    def test_path_utilities_importable_from_top_level(self):
        """Verify path utilities can be imported from top-level package."""
        from lipidetective import (
            get_project_root,
            resolve_config_path,
            resolve_data_path,
            resolve_model_path,
            resolve_output_path,
        )

        assert callable(get_project_root)
        assert callable(resolve_data_path)
        assert callable(resolve_model_path)
        assert callable(resolve_config_path)
        assert callable(resolve_output_path)

    def test_all_exports_listed_in_public_api(self):
        """Verify __all__ contains expected public API exports."""
        import lipidetective

        assert hasattr(lipidetective, "__all__")
        assert isinstance(lipidetective.__all__, list)
        assert len(lipidetective.__all__) > 0

        # Check key exports are in __all__
        expected_exports = [
            "__version__",
            "TransformerNetwork",
            "Trainer",
            "LipidLibrary",
            "get_project_root",
            "resolve_data_path",
        ]
        for export in expected_exports:
            assert export in lipidetective.__all__, f"{export} not in __all__"
