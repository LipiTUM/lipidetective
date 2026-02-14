"""
LipiDetective: Deep Learning for Lipid Identification from MS/MS Spectra.

A PyTorch-based framework using transformer architecture to identify molecular
lipid species from tandem mass spectrometry data.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("lipidetective")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"
__author__ = "Vivian Wuerf, Nikolai Koehler, Florian Molnar, Lisa Hahnefeld, Robert Gurke, Michael Witting, Josch K. Pauling"
__license__ = "BSD-3-Clause"
__url__ = "https://github.com/LipiTUM/lipidetective"

# Import main model classes
# Import helper classes and utilities
from lipidetective.helpers.lipid_library import LipidLibrary
from lipidetective.helpers.paths import (
    get_project_root,
    resolve_config_path,
    resolve_data_path,
    resolve_model_path,
    resolve_output_path,
)
from lipidetective.helpers.utils import (
    read_yaml,
    resolve_config_paths,
    set_seeds,
    write_yaml,
)
from lipidetective.models.convolutional_network import ConvolutionalNetwork
from lipidetective.models.feedforward_network import FeedForwardNetwork
from lipidetective.models.random_forest import RandomForest
from lipidetective.models.transformer_network import TransformerNetwork
from lipidetective.workflow.h5_dataset import H5Dataset
from lipidetective.workflow.lightning_module import LightningModule
from lipidetective.workflow.prediction_dataset import PredictionDataset

# Import workflow classes
from lipidetective.workflow.trainer import Trainer

# Define public API
__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__license__",
    "__url__",
    # Models
    "TransformerNetwork",
    "ConvolutionalNetwork",
    "FeedForwardNetwork",
    "RandomForest",
    # Workflow
    "Trainer",
    "H5Dataset",
    "PredictionDataset",
    "LightningModule",
    # Helpers
    "LipidLibrary",
    "read_yaml",
    "write_yaml",
    "resolve_config_paths",
    "set_seeds",
    "get_project_root",
    "resolve_data_path",
    "resolve_model_path",
    "resolve_config_path",
    "resolve_output_path",
]
