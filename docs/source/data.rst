Data Preparation
================

LipiDetective works with two file formats: **HDF5** for training and
evaluation, and **mzML** for prediction on new spectra.

Supported Formats
-----------------

HDF5 (Training & Evaluation)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

HDF5 files store pre-processed tandem mass spectra under a single group
``/all_datasets/``. Each spectrum is stored as a **dataset** (not a group)
with a 2×N array where row 0 is m/z values and row 1 is intensities.

Metadata is stored as **HDF5 attributes** on each dataset:

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Attribute
     - Type
     - Description
   * - ``lipid_species``
     - string
     - Lipid species name (empty string for unlabeled data)
   * - ``adduct``
     - string
     - Adduct type (e.g. ``[M+H]+``)
   * - ``precursor``
     - float
     - Precursor m/z value
   * - ``polarity``
     - string
     - ``"pos"`` or ``"neg"``
   * - ``scan_index``
     - int
     - Scan index from the source file
   * - ``level``
     - int
     - MS level (2 for MS/MS)
   * - ``source``
     - string
     - Source file name

Example HDF5 structure:

.. code-block:: text

   /all_datasets/
   ├── "PC 34:1 | pos | nist | 042"    # dataset: float64 array (2, N)
   │   ├── attrs["lipid_species"] = "PC 34:1"
   │   ├── attrs["adduct"] = "[M+H]+"
   │   ├── attrs["precursor"] = 760.585
   │   ├── attrs["polarity"] = "pos"
   │   └── ...
   ├── "PE 36:2 | neg | nist | 117"
   │   └── ...
   └── ...

Use ``H5Dataset`` to load HDF5 files in your training pipeline:

.. autoclass:: lipidetective.workflow.h5_dataset.H5Dataset
   :members:
   :undoc-members:

mzML (Prediction)
^^^^^^^^^^^^^^^^^^

For prediction, LipiDetective reads raw mzML files produced by mass
spectrometry instruments. The ``PredictionDataset`` handles reading,
filtering, and converting mzML spectra:

.. autoclass:: lipidetective.workflow.prediction_dataset.PredictionDataset
   :members:
   :undoc-members:

Converting mzML to HDF5
------------------------

To prepare mzML files for training, use the included conversion script:

.. code-block:: console

   $ python -m lipidetective.helpers.mzml_to_hdf5 \
       -i path/to/mzml_folder/ \
       -o path/to/output_folder/

This reads all mzML files in the input directory and produces a single HDF5
file suitable for training.

Spectrum Processing
-------------------

During loading, spectra are processed as follows:

1. **Peak selection** — The ``n_peaks`` highest-intensity peaks are retained
   (default: 30, set via ``input_embedding.n_peaks``)
2. **m/z truncation** — Peaks above ``max_mz`` are discarded
   (default: 1600, set via ``input_embedding.max_mz``)
3. **Decimal rounding** — m/z values are rounded to ``decimal_accuracy``
   decimal places (default: 1)

These parameters are configured in the ``input_embedding`` section of the
config file. See :doc:`configuration` for details.
