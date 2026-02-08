Quickstart
==========

This guide walks you through running your first lipid prediction with
LipiDetective.

1. Copy the Config Template
----------------------------

.. code-block:: console

   $ cp config/config_templates/config_transformer.yaml my_config.yaml

2. Edit the Configuration
--------------------------

Open ``my_config.yaml`` and set the file paths for your data:

.. code-block:: yaml

   model: 'transformer'

   files:
     predict_input: 'raw/my_spectra.mzML'      # Your input mzML file
     saved_model: 'lipidetective_model.pth'     # Pre-trained model
     output: 'my_first_run'                     # Output directory

   workflow:
     predict: True
     load_model: True

See :doc:`configuration` for all available options.

3. Run the Prediction
----------------------

.. code-block:: console

   $ uv run lipidetective --config my_config.yaml

4. View the Results
--------------------

Results are written to the output directory you specified. The prediction CSV
contains identified lipid species with confidence scores:

.. code-block:: text

   experiments/my_first_run/
   ├── predictions.csv       # Lipid identifications
   └── ...

See :doc:`output` for a detailed explanation of all output files.

Next Steps
----------

- :doc:`configuration` — Customize all settings
- :doc:`data` — Prepare your own training data
- :doc:`workflows` — Train, validate, test, and tune models
