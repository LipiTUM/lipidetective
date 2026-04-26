from __future__ import annotations

import math
import os
import pathlib
from typing import Any, Literal

import torch
import torch.nn as nn
from torch import Tensor

from lipidetective.helpers.utils import read_yaml


class TransformerNetwork(nn.Module):
    def __init__(self, config: dict[str, Any], output_attentions: bool = False) -> None:
        super().__init__()

        self.config = config
        self.output_attentions = output_attentions
        self.seq_length = self.config["transformer"]["output_seq_length"]

        cwd = pathlib.Path(__file__).parent.parent.resolve()
        self.tokens = read_yaml(os.path.join(cwd, "lipid_info/lipid_components_tokens.yaml"))
        out_vocab_size = len(self.tokens)

        valid_strategies = ("beam", "greedy")
        self.decode_strategy: Literal["beam", "greedy"] = config["transformer"].get(
            "decode_strategy", "beam"
        )
        if self.decode_strategy not in valid_strategies:
            raise ValueError(
                f"Invalid decode_strategy {self.decode_strategy!r}. Must be one of {valid_strategies}."
            )

        self.encoder = Encoder(config)
        self.decoder = Decoder(config, out_vocab_size)

        self.final_lin_layer = torch.nn.Linear(
            in_features=self.config["transformer"]["d_model"], out_features=out_vocab_size
        )  # tgt vocab size - 30

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor | tuple[Tensor, list[Tensor]]:
        src_padding_mask, tgt_padding_mask, nopeak_mask = self.generate_mask(src, tgt)

        encoder_output = self.encoder(src, src_padding_mask)
        decoder_output = self.decoder(
            tgt, encoder_output, nopeak_mask, tgt_padding_mask, src_padding_mask
        )

        output: Tensor = self.final_lin_layer(decoder_output)

        if self.output_attentions:
            encoder_attention = self.get_attention_layers(src, src_padding_mask)
            return output, encoder_attention
        else:
            return output

    def generate_mask(self, src: Tensor, tgt: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        src_padding_mask = src == 0
        tgt_padding_mask = tgt == 0

        seq_length = tgt.size(1)
        nopeak_mask = (
            torch.triu(torch.ones(seq_length, seq_length), diagonal=1)
            .bool()
            .type_as(tgt_padding_mask)
        )

        return src_padding_mask, tgt_padding_mask, nopeak_mask

    def predict(
        self, src: Tensor, decode_strategy: Literal["beam", "greedy"] | None = None
    ) -> Tensor:
        strategy = decode_strategy if decode_strategy is not None else self.decode_strategy

        src_padding_mask = src == 0
        tgt = (torch.zeros((src.shape[0], self.seq_length))).type_as(src).long()
        tgt[:, 0] = 1

        encoder_output = self.encoder(src, src_padding_mask)

        if strategy == "beam":
            self.beam_decode(encoder_output, tgt, src_padding_mask)
        elif strategy == "greedy":
            self.greedy_decode(encoder_output, tgt, src_padding_mask)
        else:
            raise ValueError(f"Unknown decode_strategy {strategy!r}. Must be 'beam' or 'greedy'.")

        return tgt[:, 1:]

    def predict_top_3(self, src: Tensor) -> tuple[Tensor, Tensor]:
        src_padding_mask = src == 0
        tgt = (torch.zeros((src.shape[0], self.seq_length))).type_as(src).long()
        tgt[:, 0] = 1

        encoder_output = self.encoder(src, src_padding_mask)

        nr_beams = 3
        batch_size = tgt.shape[0]

        first_output = self.decoder(
            tgt[:, :1], encoder_output, memory_padding_mask=src_padding_mask
        )
        first_output = nn.functional.log_softmax(self.final_lin_layer(first_output), dim=-1)

        first_output_values, first_output_tokens = torch.topk(first_output[:, -1], nr_beams, dim=-1)

        beam_cache = {}
        for i in range(nr_beams):
            new_tgt = (torch.zeros((batch_size, self.seq_length))).type_as(tgt).long()
            new_tgt[:, 0] = 1
            new_tgt[:, 1] = first_output_tokens[:, i]
            probabilities = torch.unsqueeze(first_output_values[:, i], dim=-1)
            beam_cache[i] = {"tokens": new_tgt, "probabilities": probabilities}

        for idx in range(2, self.seq_length):
            candidates: dict[str, list[Tensor]] = {
                "tokens": [],
                "probabilities": [],
                "beam_tokens": [],
            }

            for _beam_nr, beam in beam_cache.items():
                tgt_temp = beam["tokens"][:, :idx]

                output = self.decoder(
                    tgt_temp, encoder_output, memory_padding_mask=src_padding_mask
                )
                log_output = nn.functional.log_softmax(self.final_lin_layer(output), dim=-1)[:, -1]

                updated_probabilities = torch.add(log_output, beam["probabilities"])
                output_probs, output_tokens = torch.topk(updated_probabilities, nr_beams, dim=-1)

                candidates["tokens"].append(output_tokens)
                candidates["probabilities"].append(output_probs)
                candidates["beam_tokens"].append(tgt_temp)

            tokens = torch.cat(candidates["tokens"], dim=1)
            probabilities = torch.cat(candidates["probabilities"], dim=1)
            beam_tokens = torch.stack(candidates["beam_tokens"], dim=1)

            top3_probs, top3_idx = torch.topk(probabilities, nr_beams, dim=-1)

            top3_tokens = torch.gather(tokens, dim=1, index=top3_idx)

            # top3_idx indexes into a flat list of nr_beams*nr_beams candidates (all beams
            # concatenated). Floor-dividing by nr_beams recovers which beam each winner
            # came from. Expand repeats the beam index across the prefix length for gather.
            source_beam_idx = torch.floor_divide(top3_idx, nr_beams)
            source_beam_idx = source_beam_idx.unsqueeze(2).expand(-1, -1, idx)
            top3_beam_tokens = torch.gather(beam_tokens, dim=1, index=source_beam_idx)

            for beam_nr, beam in beam_cache.items():
                beam_tokens = top3_beam_tokens[:, beam_nr]
                tokens = top3_tokens[:, beam_nr]

                tokens_to_update = beam["tokens"]

                beam["probabilities"] = torch.unsqueeze(top3_probs[:, beam_nr], dim=-1)

                tokens_to_update[:, :idx] = beam_tokens
                tokens_to_update[:, idx] = tokens

        final_probabilities = torch.cat(
            [beam["probabilities"] for beam in beam_cache.values()], dim=1
        )
        final_tokens = torch.stack([beam["tokens"] for beam in beam_cache.values()], dim=1)

        return final_probabilities, final_tokens

    def return_encoder_embedding(self, src: Tensor) -> Tensor:
        src_padding_mask = src == 0
        encoder_output: Tensor = self.encoder(src, src_padding_mask)

        return encoder_output

    def greedy_decode(
        self, encoder_output: Tensor, tgt: Tensor, memory_padding_mask: Tensor
    ) -> None:
        for idx in range(1, self.seq_length):
            tgt_temp = tgt[:, :idx]

            output: Tensor = self.decoder(
                tgt_temp, encoder_output, memory_padding_mask=memory_padding_mask
            )
            output = self.final_lin_layer(output)
            output_tokens: Tensor = torch.argmax(output, dim=-1)
            tgt[:, idx] = output_tokens[:, -1]

    def beam_decode(self, encoder_output: Tensor, tgt: Tensor, memory_padding_mask: Tensor) -> None:
        nr_beams = 3
        batch_size = tgt.shape[0]

        first_output = self.decoder(
            tgt[:, :1], encoder_output, memory_padding_mask=memory_padding_mask
        )
        first_output = nn.functional.log_softmax(self.final_lin_layer(first_output), dim=-1)

        first_output_values, first_output_tokens = torch.topk(first_output[:, -1], nr_beams, dim=-1)

        beam_cache = {}

        # Set up the beams
        for i in range(nr_beams):
            new_tgt = (torch.zeros((batch_size, self.seq_length))).type_as(tgt).long()
            new_tgt[:, 0] = 1
            new_tgt[:, 1] = first_output_tokens[:, i]
            probabilities = torch.unsqueeze(first_output_values[:, i], dim=-1)
            beam_cache[i] = {"tokens": new_tgt, "probabilities": probabilities}

        for idx in range(2, self.seq_length):
            candidates: dict[str, list[Tensor]] = {
                "tokens": [],
                "probabilities": [],
                "beam_tokens": [],
            }

            for _beam_nr, beam in beam_cache.items():
                tgt_temp = beam["tokens"][:, :idx]

                output = self.decoder(
                    tgt_temp, encoder_output, memory_padding_mask=memory_padding_mask
                )
                log_output = nn.functional.log_softmax(self.final_lin_layer(output), dim=-1)[:, -1]

                updated_probabilities = torch.add(log_output, beam["probabilities"])
                output_probs, output_tokens = torch.topk(updated_probabilities, nr_beams, dim=-1)

                candidates["tokens"].append(output_tokens)
                candidates["probabilities"].append(output_probs)
                candidates["beam_tokens"].append(tgt_temp)

            tokens = torch.cat(candidates["tokens"], dim=1)
            probabilities = torch.cat(candidates["probabilities"], dim=1)
            beam_tokens = torch.stack(candidates["beam_tokens"], dim=1)

            top3_probs, top3_idx = torch.topk(probabilities, nr_beams, dim=-1)

            top3_tokens = torch.gather(tokens, dim=1, index=top3_idx)

            # top3_idx indexes into a flat list of nr_beams*nr_beams candidates (all beams
            # concatenated). Floor-dividing by nr_beams recovers which beam each winner
            # came from. Expand repeats the beam index across the prefix length for gather.
            source_beam_idx = torch.floor_divide(top3_idx, nr_beams)
            source_beam_idx = source_beam_idx.unsqueeze(2).expand(-1, -1, idx)
            top3_beam_tokens = torch.gather(beam_tokens, dim=1, index=source_beam_idx)

            for beam_nr, beam in beam_cache.items():
                beam_tokens = top3_beam_tokens[:, beam_nr]
                tokens = top3_tokens[:, beam_nr]

                tokens_to_update = beam["tokens"]

                beam["probabilities"] = torch.unsqueeze(top3_probs[:, beam_nr], dim=-1)

                tokens_to_update[:, :idx] = beam_tokens
                tokens_to_update[:, idx] = tokens

        final_probabilities = torch.cat(
            [beam["probabilities"] for beam in beam_cache.values()], dim=1
        )
        final_tokens = torch.stack([beam["tokens"] for beam in beam_cache.values()], dim=1)
        top_probs = torch.argmax(final_probabilities, dim=-1)
        top_probs = torch.unsqueeze(top_probs, dim=-1).expand(-1, self.seq_length)
        top_probs = torch.unsqueeze(top_probs, dim=1)

        final_output = torch.gather(final_tokens, dim=1, index=top_probs)
        final_output = torch.squeeze(final_output, dim=1)

        tgt[:, 1:] = final_output[:, 1:]

    def get_attention_layers(self, src: Tensor, src_key_padding_mask: Tensor) -> list[Tensor]:
        x = self.encoder.input_encoder(src)
        encoder_attention: list[Tensor] = []

        with torch.no_grad():
            for layer in self.encoder.transformer_encoder.layers:
                attention_weights = layer.self_attn(
                    x,
                    x,
                    x,
                    key_padding_mask=src_key_padding_mask,
                    need_weights=True,
                    average_attn_weights=False,
                )[1]
                x = layer(x, src_key_padding_mask=src_key_padding_mask)
                encoder_attention.append(attention_weights)

        return encoder_attention


class Encoder(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        self.input_encoder = PeakEncoder(config)

        layer = torch.nn.TransformerEncoderLayer(
            d_model=config["transformer"]["d_model"],
            nhead=config["transformer"]["num_heads"],
            dim_feedforward=config["transformer"]["ffn_hidden"],
            batch_first=True,
            dropout=config["transformer"]["dropout"],
        )

        self.transformer_encoder = torch.nn.TransformerEncoder(
            layer, num_layers=config["transformer"]["num_layers"], enable_nested_tensor=False
        )

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        x = self.input_encoder(x)
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        return x


class PeakEncoder(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        n_peaks = config["input_embedding"]["n_peaks"]
        d_model = config["transformer"]["d_model"]

        decimal_accuracy = config["input_embedding"]["decimal_accuracy"]
        max_mz = config["input_embedding"]["max_mz"]

        vocab_size = max_mz * 10**decimal_accuracy

        self.input_embedding = Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, n_peaks)

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_embedding(x)
        x = self.positional_encoding(x)
        return x


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, self.d_model)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embed(x)
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int = 50) -> None:
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(max_seq_len).unsqueeze(1)
        # one frequency per column pair: div_term[k] = 1 / 10000^(2k/d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        # slice div_term so cos columns match when d_model is odd
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]  # type: ignore[index]
        return x


class Decoder(nn.Module):
    def __init__(self, config: dict[str, Any], vocab_size: int) -> None:
        super().__init__()
        self.config = config

        self.name_encoder = NameEncoder(config, vocab_size)

        layer = torch.nn.TransformerDecoderLayer(
            d_model=config["transformer"]["d_model"],
            nhead=config["transformer"]["num_heads"],
            dim_feedforward=config["transformer"]["ffn_hidden"],
            batch_first=True,
            dropout=config["transformer"]["dropout"],
        )

        self.transformer_decoder = torch.nn.TransformerDecoder(
            layer, num_layers=config["transformer"]["num_layers"]
        )

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        nopeak_mask: Tensor | None = None,
        tgt_padding_mask: Tensor | None = None,
        memory_padding_mask: Tensor | None = None,
    ) -> Tensor:
        tgt_encoded: Tensor = self.name_encoder(tgt)
        x: Tensor = self.transformer_decoder(
            tgt=tgt_encoded,
            memory=memory,
            tgt_mask=nopeak_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_padding_mask,
        )
        return x


class NameEncoder(nn.Module):
    def __init__(self, config: dict[str, Any], vocab_size: int) -> None:
        super().__init__()
        self.config = config
        seq_length = config["transformer"]["output_seq_length"]
        d_model = config["transformer"]["d_model"]

        self.name_embedding = Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, seq_length)

    def forward(self, x: Tensor) -> Tensor:
        x_emb: Tensor = self.name_embedding(x)
        x_pos: Tensor = self.positional_encoding(x_emb)
        return x_pos
