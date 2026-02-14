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

    Searches upward from this file's location for a directory containing
    pyproject.toml. Raises FileNotFoundError if not found (e.g. in an
    installed wheel layout), prompting the user to set environment
    variables instead.

    Returns:
        Path to the project root directory.
    """
    current = Path(__file__).resolve().parent
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError(
        "Could not find project root (no pyproject.toml found). "
        "Set LIPIDETECTIVE_DATA_DIR, LIPIDETECTIVE_MODELS_DIR, "
        "LIPIDETECTIVE_OUTPUT_DIR, and/or LIPIDETECTIVE_CONFIG_DIR "
        "environment variables to specify paths explicitly."
    )


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

    Environment variables:
        LIPIDETECTIVE_CONFIG_DIR: Override default config directory.
    """
    config_dir = os.getenv("LIPIDETECTIVE_CONFIG_DIR")

    if config_dir:
        base = Path(config_dir)
    else:
        base = get_project_root() / "config"

    return (base / relative_path).resolve()


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
