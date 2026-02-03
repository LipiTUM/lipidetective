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
