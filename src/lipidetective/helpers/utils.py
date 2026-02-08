from __future__ import annotations

import argparse
import logging
import math
import os
import random
import traceback
from typing import Any

import numpy as np
import torch
import yaml

from lipidetective.helpers.paths import (
    is_absolute_path,
    resolve_config_path,
    resolve_data_path,
    resolve_model_path,
    resolve_output_path,
)


def resolve_config_paths(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve all file paths in a configuration dictionary.

    Paths can be:
    - Absolute paths: Used as-is
    - Relative paths: Resolved based on path type (data, model, output)

    Args:
        config: Configuration dictionary from YAML.

    Returns:
        Config with resolved absolute paths.
    """
    resolved = config.copy()

    if "files" not in resolved:
        return resolved

    files = resolved["files"].copy()
    resolved["files"] = files

    # Data paths (train, val, test, predict inputs)
    data_keys = ["train_input", "val_input", "test_input", "predict_input"]
    for key in data_keys:
        if key in files and files[key]:
            path = files[key]
            if not is_absolute_path(path):
                files[key] = str(resolve_data_path(path))

    # Model paths
    if "saved_model" in files and files["saved_model"]:
        path = files["saved_model"]
        if not is_absolute_path(path):
            files["saved_model"] = str(resolve_model_path(path))

    # Output paths
    if "output" in files and files["output"]:
        path = files["output"]
        if not is_absolute_path(path):
            files["output"] = str(resolve_output_path(path))

    # Splitting instructions (config path)
    if "splitting_instructions" in files and files["splitting_instructions"]:
        path = files["splitting_instructions"]
        if not is_absolute_path(path):
            files["splitting_instructions"] = str(resolve_config_path(path))

    return resolved


def parse_config() -> tuple[Any, argparse.Namespace]:
    parser = argparse.ArgumentParser(
        description="This script generates a deep learning model for lipid mass spectra."
    )
    parser.add_argument(
        "--config", help="path to config file containing all parameters", required=True
    )
    parser.add_argument(
        "--head_node_ip", help="ip address of head node if running tune on cluster", required=False
    )
    arguments = parser.parse_args()

    return read_yaml(arguments.config), arguments


def read_yaml(file_to_open: str) -> Any:
    try:
        with open(file_to_open) as file:
            loaded_file = yaml.safe_load(file)
        return loaded_file
    except Exception:
        traceback.print_exc()
        return None


def write_yaml(file_to_open: str, dict_to_write: dict[str, Any]) -> None:
    try:
        with open(file_to_open, "w") as file:
            yaml.dump(dict_to_write, file)
    except Exception:
        traceback.print_exc()



def set_seeds(seed: int = 42) -> None:
    torch.set_float32_matmul_precision("high")

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.enabled = False

    logging.info(f"Random seed set as {seed}")


def is_main_process() -> bool:
    """This function is only necessary when running LipiDetective using tune and makes sure that certain processes
    are only performed once per run.
    """
    return "LOCAL_RANK" not in os.environ.keys() and "NODE_RANK" not in os.environ.keys()


def truncate(values: np.ndarray, decimal_places: int = 0) -> list[float]:
    factor = 10**decimal_places
    return [(math.floor(x * factor) / factor) for x in values]


def is_lipid_class_with_slash(lipid_name: str) -> bool:
    return lipid_name.startswith(
        (
            "Cer",
            "SM",
            "EPC",
            "IPC",
            "HexCer",
            "LacCer",
            "GalCer",
            "SHexCer",
            "GlcCer",
            "SE",
            "Hex2Cer",
            "Hex3Cer",
            "FAHFA",
            "GM1",
            "GD1a",
            "GD1b",
            "GT1b",
            "GM3",
        )
    )
