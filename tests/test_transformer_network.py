"""Tests for TransformerNetwork and related classes."""

import torch

from lipidetective.models.transformer_network import (
    Decoder,
    Embedding,
    Encoder,
    PositionalEncoding,
    TransformerNetwork,
)


class TestTransformerNetworkInitialization:
    """Tests for TransformerNetwork initialization."""

    def test_initialization_creates_encoder_decoder(self, transformer_config):
        """Model should initialize encoder and decoder components."""
        model = TransformerNetwork(transformer_config)
        assert hasattr(model, "encoder")
        assert hasattr(model, "decoder")
        assert hasattr(model, "final_lin_layer")

    def test_seq_length_from_config(self, transformer_config):
        """Sequence length should be taken from config."""
        model = TransformerNetwork(transformer_config)
        assert model.seq_length == transformer_config["transformer"]["output_seq_length"]

    def test_output_attentions_flag(self, transformer_config):
        """Model should support output_attentions flag."""
        model = TransformerNetwork(transformer_config, output_attentions=True)
        assert model.output_attentions is True

        model_no_attn = TransformerNetwork(transformer_config, output_attentions=False)
        assert model_no_attn.output_attentions is False


class TestTransformerForward:
    """Tests for transformer forward pass."""

    def test_forward_returns_correct_shape(self, transformer_config):
        """Forward pass should return logits with correct shape."""
        model = TransformerNetwork(transformer_config)
        model.eval()

        batch_size = 2
        n_peaks = transformer_config["input_embedding"]["n_peaks"]
        seq_length = transformer_config["transformer"]["output_seq_length"]

        src = torch.randint(1, 1000, (batch_size, n_peaks))
        tgt = torch.randint(1, 50, (batch_size, seq_length))

        with torch.no_grad():
            output = model(src, tgt)

        # Output should be (batch, seq_length, vocab_size)
        assert output.shape[0] == batch_size
        assert output.shape[1] == seq_length
        assert output.shape[2] == model.final_lin_layer.out_features

    def test_forward_with_attention_output(self, transformer_config):
        """Forward with output_attentions=True should return attention weights."""
        model = TransformerNetwork(transformer_config, output_attentions=True)
        model.eval()

        batch_size = 2
        n_peaks = transformer_config["input_embedding"]["n_peaks"]
        seq_length = transformer_config["transformer"]["output_seq_length"]

        src = torch.randint(1, 1000, (batch_size, n_peaks))
        tgt = torch.randint(1, 50, (batch_size, seq_length))

        with torch.no_grad():
            output, attention = model(src, tgt)

        assert isinstance(attention, list)
        assert len(attention) == transformer_config["transformer"]["num_layers"]


class TestTransformerMaskGeneration:
    """Tests for mask generation."""

    def test_generate_mask_shapes(self, transformer_config):
        """Generated masks should have correct shapes."""
        model = TransformerNetwork(transformer_config)

        batch_size = 2
        n_peaks = transformer_config["input_embedding"]["n_peaks"]
        seq_length = transformer_config["transformer"]["output_seq_length"]

        src = torch.randint(0, 1000, (batch_size, n_peaks))
        tgt = torch.randint(0, 50, (batch_size, seq_length))

        src_padding_mask, tgt_padding_mask, nopeak_mask = model.generate_mask(src, tgt)

        assert src_padding_mask.shape == (batch_size, n_peaks)
        assert tgt_padding_mask.shape == (batch_size, seq_length)
        assert nopeak_mask.shape == (seq_length, seq_length)

    def test_nopeak_mask_is_causal(self, transformer_config):
        """No-peak mask should be upper triangular (causal)."""
        model = TransformerNetwork(transformer_config)

        src = torch.randint(0, 1000, (1, 10))
        tgt = torch.randint(0, 50, (1, 5))

        _, _, nopeak_mask = model.generate_mask(src, tgt)

        # Upper triangular should have True above diagonal
        assert nopeak_mask[0, 1].item() is True  # Position 0 cannot see position 1
        assert nopeak_mask[0, 0].item() is False  # Position 0 can see itself


class TestTransformerPredict:
    """Tests for prediction methods."""

    def test_predict_returns_correct_shape(self, transformer_config):
        """Predict should return sequence without SOS token."""
        model = TransformerNetwork(transformer_config)
        model.eval()

        batch_size = 2
        n_peaks = transformer_config["input_embedding"]["n_peaks"]
        seq_length = transformer_config["transformer"]["output_seq_length"]

        src = torch.randint(1, 1000, (batch_size, n_peaks))

        with torch.no_grad():
            predictions = model.predict(src)

        # Should be seq_length - 1 (excluding SOS)
        assert predictions.shape == (batch_size, seq_length - 1)

    def test_predict_greedy_returns_correct_shape(self, transformer_config):
        """Greedy predict should return same shape as beam search."""
        model = TransformerNetwork(transformer_config)
        model.eval()

        batch_size = 2
        n_peaks = transformer_config["input_embedding"]["n_peaks"]
        seq_length = transformer_config["transformer"]["output_seq_length"]

        src = torch.randint(1, 1000, (batch_size, n_peaks))

        with torch.no_grad():
            predictions = model.predict_greedy(src)

        assert predictions.shape == (batch_size, seq_length - 1)


class TestEncoder:
    """Tests for Encoder component."""

    def test_encoder_output_shape(self, transformer_config):
        """Encoder should output correct embedding dimension."""
        encoder = Encoder(transformer_config)
        encoder.eval()

        batch_size = 2
        n_peaks = transformer_config["input_embedding"]["n_peaks"]
        d_model = transformer_config["transformer"]["d_model"]

        src = torch.randint(1, 1000, (batch_size, n_peaks))
        mask = src == 0

        with torch.no_grad():
            output = encoder(src, mask)

        assert output.shape == (batch_size, n_peaks, d_model)


class TestDecoder:
    """Tests for Decoder component."""

    def test_decoder_output_shape(self, transformer_config, lipid_library):
        """Decoder should output correct embedding dimension."""
        vocab_size = lipid_library.nr_tokens
        decoder = Decoder(transformer_config, vocab_size)
        decoder.eval()

        batch_size = 2
        n_peaks = transformer_config["input_embedding"]["n_peaks"]
        seq_length = transformer_config["transformer"]["output_seq_length"]
        d_model = transformer_config["transformer"]["d_model"]

        tgt = torch.randint(1, vocab_size, (batch_size, seq_length))
        memory = torch.randn(batch_size, n_peaks, d_model)

        with torch.no_grad():
            output = decoder(tgt, memory)

        assert output.shape == (batch_size, seq_length, d_model)


class TestPositionalEncoding:
    """Tests for PositionalEncoding."""

    def test_positional_encoding_adds_to_input(self):
        """Positional encoding should add to input without changing shape."""
        d_model = 32
        max_seq_len = 50
        pe = PositionalEncoding(d_model, max_seq_len)

        batch_size = 2
        seq_len = 20
        x = torch.randn(batch_size, seq_len, d_model)

        output = pe(x)

        assert output.shape == x.shape
        # Output should be different from input (encoding added)
        assert not torch.allclose(output, x)

    def test_positional_encoding_deterministic(self):
        """Same input should produce same output (no randomness)."""
        d_model = 32
        pe = PositionalEncoding(d_model, max_seq_len=50)

        x = torch.randn(2, 10, d_model)
        output1 = pe(x.clone())
        output2 = pe(x.clone())

        assert torch.allclose(output1, output2)


class TestEmbedding:
    """Tests for Embedding layer."""

    def test_embedding_output_shape(self):
        """Embedding should output correct dimension."""
        vocab_size = 1000
        d_model = 64
        embedding = Embedding(vocab_size, d_model)

        batch_size = 2
        seq_len = 10
        x = torch.randint(0, vocab_size, (batch_size, seq_len))

        output = embedding(x)

        assert output.shape == (batch_size, seq_len, d_model)

    def test_embedding_scaled_by_sqrt_d_model(self):
        """Embedding values should be scaled by sqrt(d_model)."""
        vocab_size = 100
        d_model = 64
        embedding = Embedding(vocab_size, d_model)

        x = torch.tensor([[1]])
        output = embedding(x)

        # Get raw embedding for comparison
        raw_embedding = embedding.embed(x)

        import math

        expected = raw_embedding * math.sqrt(d_model)
        assert torch.allclose(output, expected)


class TestTransformerPredictTop3:
    """Tests for predict_top_3 method."""

    def test_predict_top_3_returns_probabilities_and_tokens(self, transformer_config):
        """predict_top_3 should return probabilities and token sequences."""
        model = TransformerNetwork(transformer_config)
        model.eval()

        batch_size = 2
        n_peaks = transformer_config["input_embedding"]["n_peaks"]

        src = torch.randint(1, 1000, (batch_size, n_peaks))

        with torch.no_grad():
            probabilities, tokens = model.predict_top_3(src)

        # Should return 3 beams
        assert probabilities.shape == (batch_size, 3)
        assert tokens.shape[0] == batch_size
        assert tokens.shape[1] == 3  # 3 beams

    def test_predict_top_3_tokens_shape(self, transformer_config):
        """Top 3 tokens should have full sequence length."""
        model = TransformerNetwork(transformer_config)
        model.eval()

        batch_size = 1
        n_peaks = transformer_config["input_embedding"]["n_peaks"]
        seq_length = transformer_config["transformer"]["output_seq_length"]

        src = torch.randint(1, 1000, (batch_size, n_peaks))

        with torch.no_grad():
            _, tokens = model.predict_top_3(src)

        assert tokens.shape == (batch_size, 3, seq_length)


class TestTransformerPredictBeamDecode:
    """Tests for predict_beam_decode method."""

    def test_predict_beam_decode_returns_correct_shape(self, transformer_config):
        """predict_beam_decode should return same shape as predict."""
        model = TransformerNetwork(transformer_config)
        model.eval()

        batch_size = 2
        n_peaks = transformer_config["input_embedding"]["n_peaks"]
        seq_length = transformer_config["transformer"]["output_seq_length"]

        src = torch.randint(1, 1000, (batch_size, n_peaks))

        with torch.no_grad():
            predictions = model.predict_beam_decode(src)

        assert predictions.shape == (batch_size, seq_length - 1)


class TestTransformerReturnEncoderEmbedding:
    """Tests for return_encoder_embedding method."""

    def test_return_encoder_embedding_shape(self, transformer_config):
        """Should return encoder output with correct dimensions."""
        model = TransformerNetwork(transformer_config)
        model.eval()

        batch_size = 2
        n_peaks = transformer_config["input_embedding"]["n_peaks"]
        d_model = transformer_config["transformer"]["d_model"]

        src = torch.randint(1, 1000, (batch_size, n_peaks))

        with torch.no_grad():
            embedding = model.return_encoder_embedding(src)

        assert embedding.shape == (batch_size, n_peaks, d_model)

    def test_return_encoder_embedding_different_inputs(self, transformer_config):
        """Different inputs should produce different embeddings."""
        model = TransformerNetwork(transformer_config)
        model.eval()

        n_peaks = transformer_config["input_embedding"]["n_peaks"]

        src1 = torch.randint(1, 500, (1, n_peaks))
        src2 = torch.randint(500, 1000, (1, n_peaks))

        with torch.no_grad():
            emb1 = model.return_encoder_embedding(src1)
            emb2 = model.return_encoder_embedding(src2)

        assert not torch.allclose(emb1, emb2)
