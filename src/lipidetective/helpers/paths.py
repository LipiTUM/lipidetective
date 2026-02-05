"""Path resolution utilities for LipiDetective.

Supports:
- Relative paths from project root
- Environment variable overrides
- Cross-platform compatibility
"""

import os
from pathlib import Path


def get_project_root() -> Path:
    """Get the project root directory.

    Returns the directory containing pyproject.toml by navigating up
    from this file's location (src/lipidetective/helpers/paths.py).

    Returns:
        Path to the project root directory.
    """
    current = Path(__file__).resolve()
    # Navigate up from src/lipidetective/helpers/paths.py
    root = current.parent.parent.parent.parent
    return root


def resolve_data_path(relative_path: str | Path) -> Path:
    """Resolve a data file path.

    Args:
        relative_path: Path relative to data directory.

    Returns:
        Absolute path to data file.

    Environment variables:
        LIPIDETECTIVE_DATA_DIR: Override default data directory.
    """
    data_dir = os.getenv("LIPIDETECTIVE_DATA_DIR")

    if data_dir:
        base = Path(data_dir)
    else:
        base = get_project_root() / "data"

    return (base / relative_path).resolve()


def resolve_model_path(relative_path: str | Path) -> Path:
    """Resolve a model file path.

    Args:
        relative_path: Path relative to models directory.

    Returns:
        Absolute path to model file.

    Environment variables:
        LIPIDETECTIVE_MODELS_DIR: Override default models directory.
    """
    models_dir = os.getenv("LIPIDETECTIVE_MODELS_DIR")

    if models_dir:
        base = Path(models_dir)
    else:
        base = get_project_root() / "models"

    return (base / relative_path).resolve()


def resolve_config_path(relative_path: str | Path) -> Path:
    """Resolve a configuration file path.

    Args:
        relative_path: Path relative to config directory.

    Returns:
        Absolute path to config file.
    """
    return (get_project_root() / "config" / relative_path).resolve()


def resolve_output_path(relative_path: str | Path) -> Path:
    """Resolve an experiment output path.

    Args:
        relative_path: Path relative to experiments directory.

    Returns:
        Absolute path for experiment output.

    Environment variables:
        LIPIDETECTIVE_OUTPUT_DIR: Override default output directory.
    """
    output_dir = os.getenv("LIPIDETECTIVE_OUTPUT_DIR")

    if output_dir:
        base = Path(output_dir)
    else:
        base = get_project_root() / "experiments"

    return (base / relative_path).resolve()


def is_absolute_path(path: str | Path) -> bool:
    """Check if a path is absolute (including ~ home paths).

    Use this to determine if a path should skip resolution.
    Relative paths should always be resolved via the resolve_*_path
    functions for consistent, CWD-independent behavior.

    Args:
        path: Path to check.

    Returns:
        True if path is absolute, False otherwise.
    """
    p = Path(path).expanduser()
    return p.is_absolute()
