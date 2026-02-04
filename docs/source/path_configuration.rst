Path Configuration
==================

LipiDetective uses flexible path resolution to support different environments.

Relative Paths
--------------

All paths in configuration files can be specified as relative paths, which are
resolved based on their type:

- **Data paths** (train_input, val_input, test_input, predict_input): relative to ``data/`` directory
- **Model paths** (saved_model): relative to ``models/`` directory
- **Output paths** (output): relative to ``experiments/`` directory
- **Config paths** (splitting_instructions): relative to ``config/`` directory

Example Configuration
---------------------

.. code-block:: yaml

    files:
      # Relative to data/
      train_input: 'processed/train_dataset.hdf5'
      val_input: 'processed/val_dataset.hdf5'
      test_input: 'processed/test_dataset.hdf5'
      predict_input: 'raw/sample.mzML'

      # Relative to models/
      saved_model: 'lipidetective_model.pth'

      # Relative to experiments/
      output: 'my_experiment'

      # Relative to config/
      splitting_instructions: 'validation_splits/train_val_split.yaml'

Absolute paths are also supported and will be used as-is.

Environment Variables
---------------------

Override default base directories using environment variables:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Variable
     - Description
   * - ``LIPIDETECTIVE_DATA_DIR``
     - Base directory for data files
   * - ``LIPIDETECTIVE_MODELS_DIR``
     - Base directory for model files
   * - ``LIPIDETECTIVE_OUTPUT_DIR``
     - Base directory for experiment outputs

Example usage:

.. code-block:: bash

    # Set custom data directory
    export LIPIDETECTIVE_DATA_DIR=/mnt/data/lipidomics

    # Run with relative paths resolved from custom directory
    lipidetective --config config/config_templates/config_transformer.yaml

Programmatic Access
-------------------

The path resolution functions can be used directly in Python:

.. code-block:: python

    from lipidetective.helpers.paths import (
        get_project_root,
        resolve_data_path,
        resolve_model_path,
        resolve_output_path,
    )

    # Get project root
    root = get_project_root()

    # Resolve paths
    data_file = resolve_data_path('processed/dataset.hdf5')
    model_file = resolve_model_path('lipidetective_model.pth')
    output_dir = resolve_output_path('experiment_001')
