from __future__ import annotations

from typing import Any

import torch.nn as nn
from torch import Tensor

from lipidetective.models.transformer_network import PeakEncoder, TransformerNetwork


class LSTMNetwork(TransformerNetwork):
    """Encoder-decoder model with an LSTM encoder and a transformer decoder.

    Replaces the transformer encoder with a bidirectional LSTM while reusing the
    full transformer decoder, beam search, and token vocabulary from TransformerNetwork.
    This enables a direct architecture comparison: the only variable is the encoder.

    TODO: LSTMNetwork inheriting from TransformerNetwork is an architectural workaround.
    The right design is an abstract EncoderDecoderNetwork base class from which both
    TransformerNetwork and LSTMNetwork inherit, each with their own config section.
    Until then, LSTMEncoder reads from config["transformer"] for shared parameters.
    """

    def __init__(self, config: dict[str, Any], output_attentions: bool = False) -> None:
        # Initialize the parent (builds decoder, final_lin_layer, tokens, etc.)
        super().__init__(config, output_attentions=False)

        # Override the encoder with an LSTM-based encoder. LSTMEncoder inherits from
        # nn.Module rather than Encoder (to avoid the unused TransformerEncoder stack),
        # so the assignment widens the declared type — intentional, see TODO above.
        self.encoder = LSTMEncoder(config)  # type: ignore[assignment]

        # Attention extraction is not supported for the LSTM encoder
        self.output_attentions = False

    def get_attention_layers(self, src: Tensor, src_key_padding_mask: Tensor) -> list[Tensor]:
        raise NotImplementedError("Attention extraction is not supported for LSTMNetwork")


class LSTMEncoder(nn.Module):
    """LSTM encoder that produces the same output shape as the transformer encoder.

    Uses the same PeakEncoder (m/z embedding + positional encoding) as the transformer,
    followed by a bidirectional LSTM. A linear projection maps the concatenated
    bidirectional hidden states back to d_model dimensions.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        d_model = config["transformer"]["d_model"]
        self.input_encoder = PeakEncoder(config)
        num_layers = config["transformer"]["num_layers"]
        dropout = config["transformer"]["dropout"] if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        # Project bidirectional output (2 * d_model) back to d_model
        self.projection = nn.Linear(2 * d_model, d_model)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        # Embed m/z tokens (same as transformer encoder)
        x = self.input_encoder(x)

        # Zero out padded positions before feeding to LSTM
        x = x * (~mask).unsqueeze(-1).float()

        # Run through bidirectional LSTM
        output: Tensor = self.lstm(x)[0]

        # Project back to d_model
        output = self.projection(output)

        # Zero out padded positions in the output
        output = output * (~mask).unsqueeze(-1).float()

        return output
