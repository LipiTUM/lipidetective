Configuration
=============

LipiDetective is configured entirely through a YAML file passed via the
``--config`` flag. A template is provided at
``config/config_templates/config_transformer.yaml``.

.. code-block:: console

   $ uv run lipidetective --config path/to/config.yaml

Model Selection
---------------

.. code-block:: yaml

   model: 'transformer'

Choose the model architecture. Options:

- ``transformer`` — Encoder-decoder transformer for sequence generation (recommended)
- ``convolutional`` — 3-layer CNN for regression
- ``feedforward`` — Fully connected network
- ``random_forest`` — Scikit-learn random forest classifier

CUDA
----

.. code-block:: yaml

   cuda:
     gpu_nr: [1]

Select which GPUs to use. Pass a list of GPU indices, e.g. ``[0, 1]`` for
multi-GPU training.

File Paths
----------

.. code-block:: yaml

   files:
     train_input: 'processed/train_dataset.hdf5'
     val_input: 'processed/val_dataset.hdf5'
     test_input: 'processed/test_dataset.hdf5'
     predict_input: 'raw/sample.mzML'
     saved_model: 'lipidetective_model.pth'
     output: 'output'
     splitting_instructions: 'validation_splits/train_val_split.yaml'

All paths can be **relative** or **absolute**. Relative paths are resolved from
default base directories:

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - Path type
     - Base directory
     - Examples
   * - Data paths (``train_input``, ``val_input``, ``test_input``, ``predict_input``)
     - ``data/``
     - HDF5 or mzML files
   * - Model paths (``saved_model``)
     - ``models/``
     - Saved model weights
   * - Output paths (``output``)
     - ``experiments/``
     - Experiment results
   * - Config paths (``splitting_instructions``)
     - ``config/``
     - Validation split definitions

Absolute paths are used as-is.

Validation Split Precedence
^^^^^^^^^^^^^^^^^^^^^^^^^^^

When training with validation (``workflow.validate: True``), the data split
strategy is determined by which fields are set, checked in this order:

1. **``val_input``** — If set, training and validation use separate HDF5 files.
   Both ``splitting_instructions`` and k-fold splitting are ignored.
2. **``splitting_instructions``** — If set (and ``val_input`` is empty), the
   referenced YAML file defines which lipid species go into the validation set.
   Data is read from ``train_input`` only.
3. **K-fold** (default) — If neither is set, ``train_input`` is split into
   ``training.k`` folds by lipid species for cross-validation.

Environment Variable Overrides
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
   * - ``LIPIDETECTIVE_CONFIG_DIR``
     - Base directory for config files (e.g. splitting instructions)

Example:

.. code-block:: bash

   export LIPIDETECTIVE_DATA_DIR=/mnt/data/lipidomics
   uv run lipidetective --config config.yaml

Programmatic Access
^^^^^^^^^^^^^^^^^^^

Path resolution functions can be used directly in Python:

.. code-block:: python

   from lipidetective import (
       get_project_root,
       resolve_data_path,
       resolve_model_path,
       resolve_output_path,
       resolve_config_path,
   )

   data_file = resolve_data_path('processed/dataset.hdf5')
   model_file = resolve_model_path('lipidetective_model.pth')

Workflow
--------

.. code-block:: yaml

   workflow:
     train: False
     validate: False
     test: False
     tune: False
     predict: True
     save_model: False
     load_model: True
     log_every_n_steps: 10

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Key
     - Default
     - Description
   * - ``train``
     - ``False``
     - Train the model
   * - ``validate``
     - ``False``
     - Enable validation during training (requires ``train: True``)
   * - ``test``
     - ``False``
     - Evaluate on the test set
   * - ``tune``
     - ``False``
     - Run hyperparameter tuning with Ray Tune
   * - ``predict``
     - ``True``
     - Run prediction on mzML input
   * - ``save_model``
     - ``False``
     - Save model weights to the output directory after training
   * - ``load_model``
     - ``True``
     - Load pre-trained model weights from ``files.saved_model``
   * - ``log_every_n_steps``
     - ``10``
     - PyTorch Lightning logging frequency

.. warning::

   When loading a pre-trained model (``load_model: True``), the
   ``transformer``, ``input_embedding``, and ``model`` settings in your config
   **must exactly match** the config used to train that model. The saved
   ``.pth`` file contains only weight tensors — no architecture metadata. If
   any parameter differs (e.g. ``d_model``, ``num_heads``, ``n_peaks``),
   PyTorch will raise a ``RuntimeError`` due to mismatched tensor shapes.

   Always keep the config file that was used for training alongside the saved
   model.

Training
--------

.. code-block:: yaml

   training:
     k: 6
     learning_rate: 0.004
     lr_step: 2
     epochs: 15
     batch: 512
     nr_workers: 0

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Key
     - Default
     - Description
   * - ``k``
     - ``6``
     - Number of folds for k-fold cross-validation
   * - ``learning_rate``
     - ``0.004``
     - Initial learning rate
   * - ``lr_step``
     - ``2``
     - Step size for learning rate scheduler
   * - ``epochs``
     - ``15``
     - Number of training epochs
   * - ``batch``
     - ``512``
     - Training batch size
   * - ``nr_workers``
     - ``0``
     - DataLoader worker processes (``0`` = main process)

Test
----

.. code-block:: yaml

   test:
     batch: 512
     confidence_score: True

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Key
     - Default
     - Description
   * - ``batch``
     - ``512``
     - Test batch size
   * - ``confidence_score``
     - ``True``
     - Compute confidence scores for predictions

Prediction
----------

.. code-block:: yaml

   predict:
     output: "best_prediction"
     batch: 512
     save_spectrum: False
     confidence_threshold: 0.98
     keep_empty: False
     keep_wrong_polarity_preds: False

.. list-table::
   :header-rows: 1
   :widths: 35 20 45

   * - Key
     - Default
     - Description
   * - ``output``
     - ``"best_prediction"``
     - ``"best_prediction"`` for top-1 or ``"top3"`` for top-3 results
   * - ``batch``
     - ``512``
     - Prediction batch size
   * - ``save_spectrum``
     - ``False``
     - Include raw spectrum data in output
   * - ``confidence_threshold``
     - ``0.98``
     - Minimum confidence to report a prediction
   * - ``keep_empty``
     - ``False``
     - Include spectra with no confident prediction
   * - ``keep_wrong_polarity_preds``
     - ``False``
     - Keep predictions with mismatched polarity

Hyperparameter Tuning
---------------------

.. code-block:: yaml

   tune:
     nr_trials: 1
     grace_period: 2
     resources_per_trial: null

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Key
     - Default
     - Description
   * - ``nr_trials``
     - ``1``
     - Number of Ray Tune trials
   * - ``grace_period``
     - ``2``
     - Minimum epochs before early stopping (ASHA scheduler)
   * - ``resources_per_trial``
     - ``null``
     - CPU/GPU allocation per trial (``null`` = auto-detect)

Input Embedding
---------------

.. code-block:: yaml

   input_embedding:
     n_peaks: 30
     max_mz: 1600
     decimal_accuracy: 1

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Key
     - Default
     - Description
   * - ``n_peaks``
     - ``30``
     - Number of highest-intensity peaks to retain
   * - ``max_mz``
     - ``1600``
     - Maximum m/z value for spectrum binning
   * - ``decimal_accuracy``
     - ``1``
     - Decimal precision for m/z values

Transformer
-----------

.. code-block:: yaml

   transformer:
     d_model: 32
     num_heads: 4
     dropout: 0.1
     ffn_hidden: 256
     num_layers: 2
     output_seq_length: 11

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Key
     - Default
     - Description
   * - ``d_model``
     - ``32``
     - Embedding dimension (must be divisible by ``num_heads``)
   * - ``num_heads``
     - ``4``
     - Number of attention heads
   * - ``dropout``
     - ``0.1``
     - Dropout rate
   * - ``ffn_hidden``
     - ``256``
     - Hidden dimension of the feed-forward layers
   * - ``num_layers``
     - ``2``
     - Number of encoder/decoder layers
   * - ``output_seq_length``
     - ``11``
     - Maximum output token sequence length

WandB Integration
-----------------

.. code-block:: yaml

   wandb:
     group: 'Debugging'

Uncomment the ``wandb`` section in the config to enable
`Weights & Biases <https://wandb.ai>`_ experiment tracking. The ``group``
field organizes runs within a WandB project.

Comment
-------

.. code-block:: yaml

   comment: 'Info on the purpose of the current run'

A free-text field to annotate the purpose of the current experiment. Logged
with the run metadata.
