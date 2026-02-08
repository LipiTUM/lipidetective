Workflows
=========

LipiDetective supports five workflows, all controlled via the ``workflow``
section of the YAML config. Multiple workflows can be enabled in a single run.

.. code-block:: yaml

   workflow:
     train: True
     validate: True
     test: False
     tune: False
     predict: False

Training
--------

Set ``workflow.train: True`` to train a model from scratch.

Training reads spectra from the HDF5 file at ``files.train_input`` and runs for
``training.epochs`` epochs with the specified learning rate and batch size.

**K-fold cross-validation** is used by default. Set ``training.k`` to control
the number of folds (default: 6). Each fold trains an independent model and
reports per-fold metrics.

To use a **custom validation split** instead of k-fold, provide a
``files.splitting_instructions`` YAML file that defines the train/validation
partition.

After training, set ``workflow.save_model: True`` to save the model weights as
``lipidetective_model.pth`` inside the experiment output directory
(``files.output``).

Validation
----------

Set ``workflow.validate: True`` to evaluate on a held-out validation set.

Validation uses the dataset at ``files.val_input``. Metrics include loss,
accuracy, and lipid-class-wise performance.

Testing
-------

Set ``workflow.test: True`` to evaluate on the test set.

Testing uses the dataset at ``files.test_input`` and produces detailed metrics
including:

- Overall accuracy
- Per-lipid-class confusion matrices
- Confidence scores (when ``test.confidence_score: True``)

Prediction
----------

Set ``workflow.predict: True`` to identify lipids in new mzML spectra.

Prediction reads mzML files from ``files.predict_input`` and outputs
identifications to the ``files.output`` directory. Key prediction settings:

- ``predict.output`` — ``"best_prediction"`` for top-1 or ``"top3"`` for top-3
- ``predict.confidence_threshold`` — Minimum confidence to report (default: 0.98)
- ``predict.keep_empty`` — Whether to include unidentified spectra
- ``predict.keep_wrong_polarity_preds`` — Whether to keep polarity-mismatched results

A pre-trained model must be loaded (``workflow.load_model: True``).

Hyperparameter Tuning
---------------------

Set ``workflow.tune: True`` to run automated hyperparameter search using
`Ray Tune <https://docs.ray.io/en/latest/tune/index.html>`_.

Tuning uses the ASHA scheduler for early stopping and supports WandB logging.
Configure via the ``tune`` section:

- ``tune.nr_trials`` — Number of hyperparameter combinations to try
- ``tune.grace_period`` — Minimum epochs before a trial can be stopped
- ``tune.resources_per_trial`` — CPU/GPU allocation (``null`` for auto-detect)

Trainer API
-----------

The ``Trainer`` class orchestrates all workflows:

.. autoclass:: lipidetective.workflow.trainer.Trainer
   :members:
   :undoc-members:
