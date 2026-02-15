from __future__ import annotations

import argparse
import logging
import math
import os
import random
from pathlib import Path
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


def setup_logging(config: dict[str, Any]) -> None:
    """Configure logging from the ``logging`` section of the YAML config.

    Reads ``config["logging"]["level"]`` (default ``"INFO"``) and
    ``config["logging"]["file"]`` (default ``None``).  Clears the bootstrap
    handlers installed by ``basicConfig`` and installs a stderr
    ``StreamHandler`` plus an optional ``FileHandler``.
    """
    log_config = (config or {}).get("logging") or {}
    level_name = str(log_config.get("level", "INFO")).upper()
    log_file: str | None = log_config.get("file")

    level = getattr(logging, level_name, None)
    if not isinstance(level, int):
        level = logging.INFO

    fmt = "%(asctime)s %(levelname)s: %(message)s"
    datefmt = "%d/%m/%Y - %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=datefmt)

    root = logging.getLogger()
    root.setLevel(level)

    # Remove bootstrap handlers set by basicConfig
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)


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
            else:
                files[key] = str(Path(path).expanduser().resolve())

    # Model paths
    for model_key in ("saved_model", "model_config"):
        if model_key in files and files[model_key]:
            path = files[model_key]
            if not is_absolute_path(path):
                files[model_key] = str(resolve_model_path(path))
            else:
                files[model_key] = str(Path(path).expanduser().resolve())

    # Output paths
    if "output" in files and files["output"]:
        path = files["output"]
        if not is_absolute_path(path):
            files["output"] = str(resolve_output_path(path))
        else:
            files["output"] = str(Path(path).expanduser().resolve())

    # Splitting instructions (config path or sentinel value like "leakage")
    splitting_sentinels = {"leakage"}
    if "splitting_instructions" in files and files["splitting_instructions"]:
        path = files["splitting_instructions"]
        if path not in splitting_sentinels:
            if not is_absolute_path(path):
                files["splitting_instructions"] = str(resolve_config_path(path))
            else:
                files["splitting_instructions"] = str(Path(path).expanduser().resolve())

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
        logging.exception(f"Failed to read YAML file: {file_to_open}")
        return None


def write_yaml(file_to_open: str, dict_to_write: dict[str, Any]) -> bool:
    try:
        with open(file_to_open, "w") as file:
            yaml.dump(dict_to_write, file)
        return True
    except Exception:
        logging.exception(f"Failed to write YAML file: {file_to_open}")
        return False


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
    # Guard against float representation error where x * factor lands just
    # below an integer (e.g. 200.0 * 10 → 1999.9999999999998).
    eps = 1e-9
    return [(math.floor(x * factor + eps) / factor) for x in values]


def extract_model_metadata(config: dict[str, Any]) -> dict[str, Any]:
    """Extract architecture-relevant config sections for a model metadata sidecar.

    Returns a dict with ``lipidetective_version``, ``model``,
    ``input_embedding``, and the model-specific section (e.g. ``transformer``).
    """
    import lipidetective

    model_type: str = config["model"]
    metadata: dict[str, Any] = {
        "lipidetective_version": lipidetective.__version__,
        "model": model_type,
        "input_embedding": config["input_embedding"],
    }
    if model_type in config:
        metadata[model_type] = config[model_type]
    return metadata


def validate_model_metadata(
    config: dict[str, Any], metadata_path: str, *, source: str = ""
) -> dict[str, Any]:
    """Validate current config against a saved model metadata sidecar.

    If the sidecar file exists, logs mismatches as warnings and merges
    the saved architecture sections into *config* (shallow auto-override).
    If the sidecar is missing, returns *config* unchanged for backwards
    compatibility.

    Args:
        config: Current configuration dictionary.
        metadata_path: Path to the model_config.yaml sidecar.
        source: How the path was resolved (e.g. ``"config"`` or
            ``"auto-detected"``), included in log messages.
    """
    source_label = f" ({source})" if source else ""
    if not Path(metadata_path).is_file():
        logging.debug("No model metadata sidecar found at: %s%s", metadata_path, source_label)
        return config

    metadata = read_yaml(metadata_path)
    if metadata is None:
        logging.warning("Could not read model metadata: %s", metadata_path)
        return config

    logging.info("Loaded model metadata%s: %s", source_label, metadata_path)

    metadata_model = metadata.get("model")
    config_model = config.get("model")
    if metadata_model and config_model and metadata_model != config_model:
        logging.warning(
            "Model type mismatch: metadata has %r but config has %r; "
            "skipping model-specific overrides.",
            metadata_model,
            config_model,
        )
        model_type = ""
    else:
        model_type = metadata_model or config_model or ""

    sections = ["input_embedding"]
    if model_type:
        sections.append(model_type)

    for section in sections:
        saved = metadata.get(section, {})
        current = config.get(section, {})
        if not isinstance(saved, dict) or not isinstance(current, dict):
            continue
        for key, saved_val in saved.items():
            current_val = current.get(key)
            if current_val != saved_val:
                logging.warning(
                    "Model metadata override: %s.%s = %r (was %r)",
                    section,
                    key,
                    saved_val,
                    current_val,
                )
        config[section] = {**current, **saved}

    return config


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
