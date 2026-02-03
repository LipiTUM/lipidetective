"""Pytest configuration and shared fixtures."""

import os
import tempfile

import pytest


@pytest.fixture
def sample_yaml_file():
    """Create a temporary YAML file for testing."""
    content = """
model: transformer
workflow:
  train: true
  validate: false
  test: false
  predict: false
  tune: false
transformer:
  d_model: 128
  num_heads: 4
  num_layers: 2
  output_seq_length: 20
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(content)
        f.flush()
        yield f.name
    os.unlink(f.name)
