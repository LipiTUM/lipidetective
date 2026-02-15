Output & Metrics
================

LipiDetective writes all results to a timestamped subdirectory created inside
``files.output`` (resolved relative to ``experiments/`` by default). The folder
is named ``LipiDetective_Output_<YYYY_MM_DD_HH_MM_SS>``, and a ``config.yaml``
copy is saved at its root.

Training Output
---------------

Training creates logger subdirectories inside the output folder:

.. code-block:: text

   experiments/<output>/LipiDetective_Output_<timestamp>/
   ├── config.yaml
   ├── lipidetective_model.pth          # When save_model: True
   ├── model_config.yaml                # Architecture metadata sidecar
   ├── custom_logger/
   │   ├── train_metrics.csv                        # Per-epoch loss & accuracy
   │   ├── train_predictions.csv                    # Per-batch predictions vs labels
   │   ├── validation_metrics.csv                   # Per-epoch validation metrics
   │   ├── validation_predictions.csv               # Validation predictions vs labels
   │   ├── plot_loss_accuracy_training.png           # Training loss & accuracy curves
   │   ├── plot_loss_training.png                    # Training loss curve
   │   ├── plot_loss_accuracy_validation.png         # Validation loss & accuracy curves
   │   ├── plot_loss_validation.png                  # Validation loss curve
   │   ├── confusion_matrix_train.csv                # Training confusion matrix
   │   ├── confusion_matrix_val.csv                  # Validation confusion matrix
   │   ├── confusion_matrix_heatmap_train.png        # Training confusion matrix heatmap
   │   ├── confusion_matrix_heatmap_validation.png   # Validation confusion matrix heatmap
   │   ├── train_lipid_metrics.csv                   # Per-lipid precision/recall/F1
   │   └── val_lipid_metrics.csv
   └── csv_logger/                                   # PyTorch Lightning CSVLogger output

With **k-fold cross-validation** (``training.k > 1``), each fold gets its own
subdirectory and plot filenames include the fold identifier:

.. code-block:: text

   custom_logger/
   ├── fold_1/
   │   ├── train_metrics.csv
   │   ├── plot_loss_accuracy_training_fold_1.png
   │   ├── confusion_matrix_heatmap_train.png
   │   └── ...
   ├── fold_2/
   │   └── ...
   └── ...

Metrics logged per epoch:

- **Loss** — Cross-entropy loss (training and validation)
- **Accuracy** — Custom lipid-aware accuracy that evaluates predicted token
  sequences against ground truth

Testing Output
--------------

Testing writes to ``custom_logger/`` (no fold subdirectory):

.. code-block:: text

   custom_logger/
   ├── test_metrics.csv                      # Per-step loss & accuracy
   ├── test_predictions.csv                  # Predictions vs labels
   ├── confusion_matrix_test.csv             # Confusion matrix
   ├── confusion_matrix_heatmap_testing.png  # Confusion matrix heatmap
   └── test_lipid_metrics.csv                # Per-lipid precision/recall/F1

Confusion matrices show per-lipid-class performance, helping identify which
lipid classes the model confuses. The ``test_lipid_metrics.csv`` reports
precision, recall, and F1 per lipid species.

Prediction Output
-----------------

Prediction writes directly to the output folder root (not inside a logger
subdirectory):

.. code-block:: text

   experiments/<output>/LipiDetective_Output_<timestamp>/
   ├── config.yaml
   └── predictions.csv

The ``predictions.csv`` file contains one row per identified spectrum with the
following columns:

- **file** — Source mzML file name
- **polarity** — Ion polarity of the spectrum
- **spectrum_index** — Index of the spectrum within the file
- **precursor** — Precursor m/z value
- **prediction** — Predicted lipid nomenclature
- **confidence** — Model confidence score (0–1)

Spectra below the ``predict.confidence_threshold`` are omitted by default
(set ``predict.keep_empty: True`` to include them).

When ``predict.output`` is set to ``"top3"``, an additional
``top3_predictions.csv`` is written with columns:

- **file** — Source mzML file name
- **spectrum_index** — Index of the spectrum within the file
- **prediction_1**, **confidence_1** — Top prediction and its confidence
- **prediction_2**, **confidence_2** — Second prediction
- **prediction_3**, **confidence_3** — Third prediction

Tuning Output
-------------

Hyperparameter tuning (Ray Tune) writes trial results to the output folder.
Each trial generates training and validation plots with a trial identifier:

.. code-block:: text

   experiments/<output>/LipiDetective_Output_<timestamp>/
   ├── plot_loss_accuracy_training_trial_0.png
   ├── plot_loss_accuracy_validation_trial_0.png
   ├── tune_result.txt
   └── ...

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
