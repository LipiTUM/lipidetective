Workflows
=========

LipiDetective supports four workflows, all controlled via the ``workflow``
section of the YAML config. **Only one workflow runs per invocation.** If
multiple flags are set to ``True``, the first match wins in this order:

1. ``tune``
2. ``train`` (with or without ``validate``)
3. ``test``
4. ``predict``

The ``validate`` flag is not a standalone workflow — it controls whether
training includes validation (k-fold or custom split).

.. note::

   When ``model`` is set to ``random_forest``, the workflow flags are ignored
   entirely. The random forest runs its own self-contained pipeline via
   ``run_random_forest()``.

.. code-block:: yaml

   workflow:
     train: True
     validate: True    # not a standalone workflow, modifies train
     test: False
     tune: False
     predict: False

Training
--------

Set ``workflow.train: True`` to train a model from scratch.

Training reads spectra from the HDF5 file at ``files.train_input`` and runs for
``training.epochs`` epochs with the specified learning rate and batch size.

When ``workflow.validate: True``, the validation split strategy depends on
which config fields are set (checked in this order):

1. **Separate validation file** — If ``files.val_input`` is set, it is used as
   the validation set directly. Both ``splitting_instructions`` and k-fold
   splitting are ignored.
2. **Custom split** — If ``files.splitting_instructions`` is set (and
   ``val_input`` is empty), the referenced YAML file defines which lipid
   species go into the validation set. Data is read from ``train_input`` only.
3. **K-fold cross-validation** (default) — If neither is set, ``train_input``
   is split into ``training.k`` folds (default: 6) by lipid species. Each fold
   trains an independent model and reports per-fold metrics.

After training, set ``workflow.save_model: True`` to save model weights.
A ``model_config.yaml`` sidecar is saved alongside each ``.pth`` file,
recording the architecture config used during training. The save location
depends on the training mode:

- **Single-split training** (no validation or custom split):
  ``<files.output>/LipiDetective_Output_<timestamp>/lipidetective_model.pth``
- **K-fold cross-validation**: one model per fold, e.g.
  ``<files.output>/LipiDetective_Output_<timestamp>/fold_1/lipidetective_model.pth``

When loading a model later, the sidecar is auto-detected and used to validate
the current config against the saved architecture (see :doc:`configuration`).

Validation
----------

Validation is not a standalone workflow — it is enabled by setting
``workflow.validate: True`` **together with** ``workflow.train: True``.
Setting ``validate: True`` without ``train: True`` has no effect.

The validation data source depends on the split strategy described above
(``val_input`` → ``splitting_instructions`` → k-fold). Metrics include loss,
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
