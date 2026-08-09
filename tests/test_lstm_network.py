"""Tests for LSTMNetwork and LSTMEncoder."""

import pytest
import torch

from lipidetective.models.lstm_network import LSTMEncoder, LSTMNetwork


class TestLSTMNetworkInitialization:
    """Tests for LSTMNetwork initialization."""

    def test_encoder_is_lstm(self, transformer_config):
        """Encoder should be replaced with LSTMEncoder, not the default Encoder."""
        model = LSTMNetwork(transformer_config)
        assert isinstance(model.encoder, LSTMEncoder)

    def test_has_decoder_and_output_layer(self, transformer_config):
        """Decoder and final linear layer should be inherited from TransformerNetwork."""
        model = LSTMNetwork(transformer_config)
        assert hasattr(model, "decoder")
        assert hasattr(model, "final_lin_layer")

    def test_output_attentions_always_false(self, transformer_config):
        """LSTM encoder does not support attention extraction."""
        model = LSTMNetwork(transformer_config, output_attentions=True)
        assert model.output_attentions is False

    def test_lstm_encoder_has_projection(self, transformer_config):
        """LSTMEncoder should have a projection layer to map bidirectional output to d_model."""
        model = LSTMNetwork(transformer_config)
        d_model = transformer_config["transformer"]["d_model"]
        assert model.encoder.projection.in_features == 2 * d_model
        assert model.encoder.projection.out_features == d_model


class TestLSTMEncoderOutputShape:
    """Tests for LSTMEncoder output dimensions."""

    def test_encoder_output_shape_matches_transformer(self, transformer_config):
        """LSTMEncoder should produce the same output shape as the transformer encoder."""
        encoder = LSTMEncoder(transformer_config)
        encoder.eval()

        batch_size = 2
        n_peaks = transformer_config["input_embedding"]["n_peaks"]
        d_model = transformer_config["transformer"]["d_model"]

        src = torch.randint(1, 1000, (batch_size, n_peaks))
        mask = src == 0

        with torch.no_grad():
            output = encoder(src, mask)

        assert output.shape == (batch_size, n_peaks, d_model)

    def test_encoder_zeros_padded_positions(self, transformer_config):
        """Padded positions (mask=True) should be zeroed out in encoder output."""
        encoder = LSTMEncoder(transformer_config)
        encoder.eval()

        n_peaks = transformer_config["input_embedding"]["n_peaks"]

        src = torch.randint(1, 1000, (1, n_peaks))
        mask = torch.zeros(1, n_peaks, dtype=torch.bool)
        mask[0, -5:] = True  # mask last 5 positions

        with torch.no_grad():
            output = encoder(src, mask)

        assert torch.all(output[0, -5:] == 0)

    def test_real_token_outputs_invariant_to_padding_length(self, transformer_config):
        """Real-token outputs must not depend on how much padding follows them.

        Regression test for the packed-sequence fix: without pack_padded_sequence,
        the bidirectional LSTM's backward pass reads padding tokens before reaching
        real positions, so real-token outputs drift depending on trailing padding.
        """
        encoder = LSTMEncoder(transformer_config)
        encoder.eval()

        n_peaks = transformer_config["input_embedding"]["n_peaks"]
        real_len = 3
        assert real_len + 2 <= n_peaks, "fixture n_peaks too small for this test"

        torch.manual_seed(0)
        real_tokens = torch.randint(1, 1000, (1, real_len))

        def run(total_len: int) -> torch.Tensor:
            src = torch.zeros(1, total_len, dtype=torch.long)
            src[0, :real_len] = real_tokens
            mask = torch.zeros(1, total_len, dtype=torch.bool)
            mask[0, real_len:] = True
            with torch.no_grad():
                return encoder(src, mask)

        output_short_pad = run(real_len + 1)
        output_long_pad = run(n_peaks)

        assert torch.allclose(
            output_short_pad[0, :real_len], output_long_pad[0, :real_len], atol=1e-6
        )


class TestLSTMNetworkForward:
    """Tests for LSTMNetwork forward pass."""

    def test_forward_returns_correct_shape(self, transformer_config):
        """Forward pass should return logits with correct shape."""
        model = LSTMNetwork(transformer_config)
        model.eval()

        batch_size = 2
        n_peaks = transformer_config["input_embedding"]["n_peaks"]
        seq_length = transformer_config["transformer"]["output_seq_length"]

        src = torch.randint(1, 1000, (batch_size, n_peaks))
        tgt = torch.randint(1, 50, (batch_size, seq_length))

        with torch.no_grad():
            output = model(src, tgt)

        assert output.shape[0] == batch_size
        assert output.shape[1] == seq_length
        assert output.shape[2] == model.final_lin_layer.out_features


class TestLSTMNetworkPredict:
    """Tests for inherited prediction methods."""

    def test_predict_returns_correct_shape(self, transformer_config):
        """predict() should return sequence of correct length."""
        model = LSTMNetwork(transformer_config)
        model.eval()

        batch_size = 2
        n_peaks = transformer_config["input_embedding"]["n_peaks"]
        seq_length = transformer_config["transformer"]["output_seq_length"]

        src = torch.randint(1, 1000, (batch_size, n_peaks))

        with torch.no_grad():
            predictions = model.predict(src)

        assert predictions.shape == (batch_size, seq_length - 1)

    def test_predict_greedy_returns_correct_shape(self, transformer_config):
        """Greedy strategy should return same shape as beam search predict."""
        model = LSTMNetwork(transformer_config)
        model.eval()

        batch_size = 2
        n_peaks = transformer_config["input_embedding"]["n_peaks"]
        seq_length = transformer_config["transformer"]["output_seq_length"]

        src = torch.randint(1, 1000, (batch_size, n_peaks))

        with torch.no_grad():
            predictions = model.predict(src, decode_strategy="greedy")

        assert predictions.shape == (batch_size, seq_length - 1)

    def test_get_attention_layers_raises(self, transformer_config):
        """Attention extraction should raise NotImplementedError for LSTM encoder."""
        model = LSTMNetwork(transformer_config)

        src = torch.randint(1, 1000, (1, transformer_config["input_embedding"]["n_peaks"]))
        mask = src == 0

        with pytest.raises(NotImplementedError):
            model.get_attention_layers(src, mask)
