Output & Metrics
================

LipiDetective writes all results to the directory specified by ``files.output``
(resolved relative to ``experiments/`` by default).

Output Directory Structure
--------------------------

A typical experiment produces:

.. code-block:: text

   experiments/<output>/
   ├── predictions.csv                         # Prediction results
   ├── plot_loss_accuracy_training_trial_0.png  # Training curves
   ├── plot_loss_accuracy_val_trial_0.png       # Validation curves
   ├── confusion_matrix.png                     # Class-level confusion matrix
   ├── metrics.csv                              # Per-epoch training metrics
   └── config.yaml                              # Copy of the run config

Training Metrics
----------------

During training, the following metrics are logged per epoch:

- **Loss** — Cross-entropy loss (training and validation)
- **Accuracy** — Custom lipid-aware accuracy that evaluates predicted token
  sequences against ground truth

Loss and accuracy curves are saved as PNG plots for each fold/trial.

Confusion Matrices
------------------

After testing, a confusion matrix shows per-lipid-class performance. This
helps identify which lipid classes the model confuses.

Prediction Output
-----------------

The prediction CSV contains one row per identified spectrum:

- **file** — Source mzML file name
- **scan** — Scan number within the file
- **precursor_mz** — Precursor m/z value
- **prediction** — Predicted lipid nomenclature
- **confidence** — Model confidence score (0–1)

Spectra below the ``predict.confidence_threshold`` are omitted by default
(set ``predict.keep_empty: True`` to include them).

When ``predict.output`` is set to ``"top3"``, the three most likely
predictions are reported per spectrum.

WandB Integration
-----------------

When the ``wandb`` config section is enabled, all metrics are additionally
logged to `Weights & Biases <https://wandb.ai>`_ for interactive visualization
and experiment comparison. Runs are organized by the ``wandb.group`` field.

Custom Evaluation
-----------------

The ``Evaluator`` class provides lipid-aware evaluation logic:

.. autoclass:: lipidetective.helpers.logging.Evaluator
   :members:
   :undoc-members:
