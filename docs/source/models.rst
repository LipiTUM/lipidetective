Models
======

LipiDetective supports four model architectures. Select the model via the
``model`` field in the config:

.. code-block:: yaml

   model: 'transformer'   # or 'convolutional', 'feedforward', 'random_forest'

Transformer (Recommended)
-------------------------

The primary model. Uses an encoder-decoder transformer architecture to generate
lipid nomenclature as a token sequence from an input spectrum. The encoder
processes the spectrum embedding, and the decoder autoregressively predicts
lipid tokens (headgroup, fatty acid chains, etc.).

Configure via the ``transformer`` section:

.. code-block:: yaml

   transformer:
     d_model: 32       # Embedding dimension (must be divisible by num_heads)
     num_heads: 4      # Attention heads
     dropout: 0.1
     ffn_hidden: 256   # Feed-forward hidden dimension
     num_layers: 2     # Encoder/decoder layers
     output_seq_length: 11

.. autoclass:: lipidetective.models.transformer_network.TransformerNetwork
   :members:
   :undoc-members:

Convolutional Neural Network
-----------------------------

A 3-layer CNN for regression tasks on spectral data. Useful as a baseline or
for simpler prediction tasks.

.. autoclass:: lipidetective.models.convolutional_network.ConvolutionalNetwork
   :members:
   :undoc-members:

Feed-Forward Network
--------------------

A simple fully connected network. Serves as a minimal baseline architecture.

.. autoclass:: lipidetective.models.feedforward_network.FeedForwardNetwork
   :members:
   :undoc-members:

Random Forest
-------------

A scikit-learn ``RandomForestClassifier`` wrapper. Operates outside the
PyTorch Lightning pipeline and handles its own data loading from HDF5 files.
Useful for comparison against deep learning approaches.

.. autoclass:: lipidetective.models.random_forest.RandomForest
   :members:
   :undoc-members:
