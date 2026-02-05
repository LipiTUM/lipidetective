"""Tests for the path resolution utilities."""

from pathlib import Path

from lipidetective.helpers.paths import (
    get_project_root,
    is_absolute_path,
    resolve_config_path,
    resolve_data_path,
    resolve_model_path,
    resolve_output_path,
)


class TestGetProjectRoot:
    """Tests for get_project_root function."""

    def test_returns_path_object(self):
        """Should return a Path object."""
        root = get_project_root()
        assert isinstance(root, Path)

    def test_root_contains_pyproject(self):
        """Project root should contain pyproject.toml."""
        root = get_project_root()
        assert (root / "pyproject.toml").exists()

    def test_root_contains_src(self):
        """Project root should contain src directory."""
        root = get_project_root()
        assert (root / "src" / "lipidetective").exists()


class TestResolveDataPath:
    """Tests for resolve_data_path function."""

    def test_returns_absolute_path(self, monkeypatch):
        """Should return an absolute path."""
        monkeypatch.delenv("LIPIDETECTIVE_DATA_DIR", raising=False)
        path = resolve_data_path("test.hdf5")
        assert path.is_absolute()

    def test_includes_data_directory(self, monkeypatch):
        """Path should include data directory."""
        monkeypatch.delenv("LIPIDETECTIVE_DATA_DIR", raising=False)
        path = resolve_data_path("test.hdf5")
        assert "data" in str(path)

    def test_preserves_subdirectories(self, monkeypatch):
        """Should preserve subdirectory structure."""
        monkeypatch.delenv("LIPIDETECTIVE_DATA_DIR", raising=False)
        path = resolve_data_path("processed/train.hdf5")
        assert path.name == "train.hdf5"
        assert path.parent.name == "processed"

    def test_env_override(self, monkeypatch, tmp_path):
        """Should use LIPIDETECTIVE_DATA_DIR when set."""
        custom_dir = tmp_path / "custom_data"
        custom_dir.mkdir()
        monkeypatch.setenv("LIPIDETECTIVE_DATA_DIR", str(custom_dir))

        path = resolve_data_path("test.hdf5")
        assert path.parent == custom_dir


class TestResolveModelPath:
    """Tests for resolve_model_path function."""

    def test_returns_absolute_path(self, monkeypatch):
        """Should return an absolute path."""
        monkeypatch.delenv("LIPIDETECTIVE_MODELS_DIR", raising=False)
        path = resolve_model_path("model.pth")
        assert path.is_absolute()

    def test_includes_models_directory(self, monkeypatch):
        """Path should include models directory."""
        monkeypatch.delenv("LIPIDETECTIVE_MODELS_DIR", raising=False)
        path = resolve_model_path("model.pth")
        assert "models" in str(path)

    def test_env_override(self, monkeypatch, tmp_path):
        """Should use LIPIDETECTIVE_MODELS_DIR when set."""
        custom_dir = tmp_path / "custom_models"
        custom_dir.mkdir()
        monkeypatch.setenv("LIPIDETECTIVE_MODELS_DIR", str(custom_dir))

        path = resolve_model_path("model.pth")
        assert path.parent == custom_dir


class TestResolveOutputPath:
    """Tests for resolve_output_path function."""

    def test_returns_absolute_path(self, monkeypatch):
        """Should return an absolute path."""
        monkeypatch.delenv("LIPIDETECTIVE_OUTPUT_DIR", raising=False)
        path = resolve_output_path("experiment_001")
        assert path.is_absolute()

    def test_includes_experiments_directory(self, monkeypatch):
        """Path should include experiments directory."""
        monkeypatch.delenv("LIPIDETECTIVE_OUTPUT_DIR", raising=False)
        path = resolve_output_path("experiment_001")
        assert "experiments" in str(path)

    def test_env_override(self, monkeypatch, tmp_path):
        """Should use LIPIDETECTIVE_OUTPUT_DIR when set."""
        custom_dir = tmp_path / "custom_output"
        custom_dir.mkdir()
        monkeypatch.setenv("LIPIDETECTIVE_OUTPUT_DIR", str(custom_dir))

        path = resolve_output_path("experiment_001")
        assert path.parent == custom_dir


class TestResolveConfigPath:
    """Tests for resolve_config_path function."""

    def test_returns_absolute_path(self):
        """Should return an absolute path."""
        path = resolve_config_path("config_transformer.yaml")
        assert path.is_absolute()

    def test_includes_config_directory(self):
        """Path should include config directory."""
        path = resolve_config_path("config_transformer.yaml")
        assert "config" in str(path)

    def test_preserves_subdirectories(self):
        """Should preserve subdirectory structure."""
        path = resolve_config_path("config_templates/config_transformer.yaml")
        assert path.name == "config_transformer.yaml"
        assert path.parent.name == "config_templates"


class TestIsAbsolutePath:
    """Tests for is_absolute_path function."""

    def test_absolute_path_returns_true(self):
        """Absolute paths should return True."""
        assert is_absolute_path("/some/absolute/path")

    def test_home_path_returns_true(self):
        """Home-relative paths (~) should return True."""
        assert is_absolute_path("~/some/path")
        assert is_absolute_path("~")

    def test_relative_path_returns_false(self):
        """Relative paths should return False, regardless of existence."""
        assert not is_absolute_path("relative/path")
        assert not is_absolute_path("file.txt")

    def test_existing_relative_path_returns_false(self, tmp_path, monkeypatch):
        """Existing relative paths should still return False (CWD-independent)."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "existing_file.txt").touch()
        assert not is_absolute_path("existing_file.txt")
