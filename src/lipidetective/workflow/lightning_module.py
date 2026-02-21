from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torchmetrics.aggregation import CatMetric
from torchmetrics.regression import MeanAbsoluteError, R2Score
from torchmetrics.wrappers import ClasswiseWrapper

from lipidetective.helpers.logging import CustomAccuracy, Evaluator
from lipidetective.helpers.utils import (
    extract_model_metadata,
    validate_model_metadata,
    write_yaml,
)
from lipidetective.models.convolutional_network import ConvolutionalNetwork
from lipidetective.models.feedforward_network import FeedForwardNetwork
from lipidetective.models.lstm_network import LSTMNetwork
from lipidetective.models.transformer_network import TransformerNetwork


class LightningModule(pl.LightningModule):
    def __init__(
        self,
        config: dict[str, Any],
        evaluator: Evaluator,
        trainset_lipids: list[str] | None = None,
        valset_lipids: list[str] | None = None,
        testset_lipids: list[str] | None = None,
    ) -> None:
        super().__init__()

        self.config: dict[str, Any] = config
        self.batch_size: int = self.config["training"]["batch"]
        self.nr_epochs: int = self.config["training"]["epochs"] - 1

        if self.config["workflow"]["load_model"]:
            if self.config["files"].get("model_config"):
                metadata_path = self.config["files"]["model_config"]
                metadata_source = "config"
            else:
                metadata_path = str(
                    Path(self.config["files"]["saved_model"]).parent / "model_config.yaml"
                )
                metadata_source = "auto-detected"
            self.config = validate_model_metadata(
                self.config, metadata_path, source=metadata_source
            )

        self.model: nn.Module = self.get_neural_network()

        if self.config["workflow"]["load_model"]:
            model_file_path: str = self.config["files"]["saved_model"]
            self.model.load_state_dict(torch.load(model_file_path))
            logging.info(f"Loaded pre-trained model from: {model_file_path}")

        # Declare all conditionally initialized attributes with Optional types
        self.train_mae_hg: MeanAbsoluteError | None = None
        self.train_mae_fa1: MeanAbsoluteError | None = None
        self.train_mae_fa2: MeanAbsoluteError | None = None
        self.train_r2: R2Score | None = None
        self.val_mae_hg: MeanAbsoluteError | None = None
        self.val_mae_fa1: MeanAbsoluteError | None = None
        self.val_mae_fa2: MeanAbsoluteError | None = None
        self.val_r2: R2Score | None = None

        self.trainset_names: list[str] | None = None
        self.train_custom_accuracy: ClasswiseWrapper | None = None
        self.train_predictions: CatMetric | None = None

        self.valset_names: list[str] | None = None
        self.val_custom_accuracy: ClasswiseWrapper | None = None
        self.val_predictions: CatMetric | None = None

        self.testset_names: list[str] | None = None
        self.test_custom_accuracy: ClasswiseWrapper | None = None
        self.test_predictions: CatMetric | None = None

        # Create metrics for regression models
        if not isinstance(self.model, TransformerNetwork):
            self.train_mae_hg = MeanAbsoluteError()
            self.train_mae_fa1 = MeanAbsoluteError()
            self.train_mae_fa2 = MeanAbsoluteError()
            self.train_r2 = R2Score(multioutput="uniform_average")

            if valset_lipids is not None:
                self.val_mae_hg = MeanAbsoluteError()
                self.val_mae_fa1 = MeanAbsoluteError()
                self.val_mae_fa2 = MeanAbsoluteError()
                self.val_r2 = R2Score(multioutput="uniform_average")

        if trainset_lipids is not None:
            self.trainset_names = trainset_lipids
            self.train_custom_accuracy = ClasswiseWrapper(
                CustomAccuracy(evaluator, trainset_lipids),
                labels=["train_accuracy", "train_mean_accuracy"],
            )
            self.train_predictions = CatMetric()

        if valset_lipids is not None:
            self.valset_names = valset_lipids
            self.val_custom_accuracy = ClasswiseWrapper(
                CustomAccuracy(evaluator, valset_lipids),
                labels=["val_accuracy", "val_mean_accuracy"],
            )
            self.val_predictions = CatMetric()

        if testset_lipids is not None:
            self.testset_names = testset_lipids
            self.test_custom_accuracy = ClasswiseWrapper(
                CustomAccuracy(evaluator, testset_lipids),
                labels=["test_accuracy", "test_mean_accuracy"],
            )
            self.test_predictions = CatMetric()

    def _get_custom_logger(self) -> Any:
        """Return the custom logger if active, else None.

        Handles both single-logger and multi-logger (LoggerCollection) cases
        by iterating self.loggers rather than checking only self.logger.
        """
        for logger in self.loggers:
            if logger.name == "custom_logger":
                return logger
        return None

    def configure_optimizers(self) -> Any:
        optimizer = optim.Adam(self.model.parameters(), lr=self.config["training"]["learning_rate"])
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=self.config["training"]["lr_step"], gamma=0.9
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        features = batch["features"]
        labels = batch["label"]
        lipid_info = batch["info"]
        dataset_path = batch["dataset_path"]

        if isinstance(self.model, TransformerNetwork):
            tgt_input = labels[:, :-1]
            tgt_expected = labels[:, 1:].long()

            output = self.model(features, tgt_input)

            # output_tokens = torch.argmax(output, dim=2)
            # is_last_epoch = self.current_epoch == self.nr_epochs

            # assert self.train_custom_accuracy is not None
            # accuracy_dict = self.train_custom_accuracy(
            #     output_tokens, tgt_expected, "transformer", is_last_epoch
            # )

            # if (logger := self._get_custom_logger()) is not None:
            #     preds_vs_labels = self.get_preds_vs_labels(
            #         batch_idx, output_tokens, tgt_expected, dataset_path
            #     )
            #     assert self.train_predictions is not None
            #     self.train_predictions(preds_vs_labels)
            #     logger.log_predictions(self.train_predictions, batch_idx, "train")

            output = torch.transpose(output, 1, 2)
            loss = nn.functional.cross_entropy(output, tgt_expected)

            self.log(
                "train_loss",
                loss.item(),
                on_step=True,
                on_epoch=True,
                sync_dist=True,
                batch_size=self.batch_size,
            )
            # self.log_dict(
            #     accuracy_dict,
            #     on_step=True,
            #     on_epoch=True,
            #     sync_dist=True,
            #     batch_size=self.batch_size,
            # )

            # if "wandb" in self.config and not self.config["workflow"]["tune"]:
            #     wandb.log(
            #         {
            #             "train_loss": loss.item(),
            #             "train_accuracy": accuracy_dict["customaccuracy_train_accuracy"],
            #             "train_mean_accuracy": accuracy_dict["customaccuracy_train_mean_accuracy"],
            #         }
            #     )

        else:
            output = self.model(features)
            loss = nn.functional.mse_loss(output, labels)

            assert self.train_custom_accuracy is not None
            assert self.train_mae_hg is not None
            assert self.train_mae_fa1 is not None
            assert self.train_mae_fa2 is not None

            accuracy_dict = self.train_custom_accuracy(output, lipid_info, "regression")
            self.train_mae_hg(output[:, 0], labels[:, 0])
            self.train_mae_fa1(output[:, 1], labels[:, 1])
            self.train_mae_fa2(output[:, 2], labels[:, 2])

            if output.size()[0] > 1:
                assert self.train_r2 is not None
                self.train_r2(output, labels)
                self.log(
                    "train_r2",
                    self.train_r2,
                    on_step=True,
                    on_epoch=True,
                    batch_size=self.batch_size,
                )

            self.log(
                "train_loss",
                loss.item(),
                on_step=True,
                on_epoch=True,
                sync_dist=True,
                batch_size=self.batch_size,
            )
            self.log_dict(
                accuracy_dict,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
                batch_size=self.batch_size,
            )
            self.log(
                "train_mae_hg",
                self.train_mae_hg,
                on_step=True,
                on_epoch=True,
                batch_size=self.batch_size,
            )
            self.log(
                "train_mae_fa1",
                self.train_mae_fa1,
                on_step=True,
                on_epoch=True,
                batch_size=self.batch_size,
            )
            self.log(
                "train_mae_fa2",
                self.train_mae_fa2,
                on_step=True,
                on_epoch=True,
                batch_size=self.batch_size,
            )

            if (logger := self._get_custom_logger()) is not None:
                preds_vs_labels = self.get_preds_vs_labels(batch_idx, output, labels, dataset_path)
                assert self.train_predictions is not None
                self.train_predictions(preds_vs_labels)
                logger.log_predictions(self.train_predictions, batch_idx, "train")

        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        features = batch["features"]
        labels = batch["label"]
        lipid_info = batch["info"]
        dataset_path = batch["dataset_path"]

        if isinstance(self.model, TransformerNetwork):
            tgt_input = labels[:, :-1]
            tgt_expected = labels[:, 1:].long()

            output = self.model(features, tgt_input)
            output_tokens = self.model.predict(features)

            is_last_epoch = self.current_epoch == self.nr_epochs
            assert self.val_custom_accuracy is not None
            accuracy_dict = self.val_custom_accuracy(
                output_tokens, tgt_expected, "transformer", is_last_epoch
            )

            if (logger := self._get_custom_logger()) is not None:
                preds_vs_labels = self.get_preds_vs_labels(
                    batch_idx, output_tokens, tgt_expected, dataset_path
                )
                assert self.val_predictions is not None
                self.val_predictions(preds_vs_labels)
                logger.log_predictions(self.val_predictions, batch_idx, "val")

            output = torch.transpose(output, 1, 2)
            loss = nn.functional.cross_entropy(output, tgt_expected)

            self.log(
                "val_loss",
                loss.item(),
                on_step=True,
                on_epoch=True,
                sync_dist=True,
                batch_size=self.batch_size,
            )
            self.log_dict(
                accuracy_dict,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
                batch_size=self.batch_size,
            )
            if "wandb" in self.config and not self.config["workflow"]["tune"]:
                wandb.log(
                    {
                        "val_loss": loss.item(),
                        "val_accuracy": accuracy_dict["customaccuracy_val_accuracy"],
                        "val_mean_accuracy": accuracy_dict["customaccuracy_val_mean_accuracy"],
                    }
                )

        else:
            output = self.model(features)
            loss = nn.functional.mse_loss(output, labels)

            assert self.val_custom_accuracy is not None
            assert self.val_mae_hg is not None
            assert self.val_mae_fa1 is not None
            assert self.val_mae_fa2 is not None

            accuracy_dict = self.val_custom_accuracy(output, lipid_info, "regression")
            self.val_mae_hg(output[:, 0], labels[:, 0])
            self.val_mae_fa1(output[:, 1], labels[:, 1])
            self.val_mae_fa2(output[:, 2], labels[:, 2])

            if output.size()[0] > 1:
                assert self.val_r2 is not None
                self.val_r2(output, labels)
                self.log(
                    "val_r2", self.val_r2, on_step=True, on_epoch=True, batch_size=self.batch_size
                )

            self.log(
                "val_loss",
                loss.item(),
                on_step=True,
                on_epoch=True,
                sync_dist=True,
                batch_size=self.batch_size,
            )
            self.log_dict(
                accuracy_dict,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
                batch_size=self.batch_size,
            )
            self.log(
                "val_mae_hg",
                self.val_mae_hg,
                on_step=True,
                on_epoch=True,
                batch_size=self.batch_size,
            )
            self.log(
                "val_mae_fa1",
                self.val_mae_fa1,
                on_step=True,
                on_epoch=True,
                batch_size=self.batch_size,
            )
            self.log(
                "val_mae_fa2",
                self.val_mae_fa2,
                on_step=True,
                on_epoch=True,
                batch_size=self.batch_size,
            )

            if (logger := self._get_custom_logger()) is not None:
                preds_vs_labels = self.get_preds_vs_labels(batch_idx, output, labels, dataset_path)
                assert self.val_predictions is not None
                self.val_predictions(preds_vs_labels)
                logger.log_predictions(self.val_predictions, batch_idx, "val")

        return loss

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        features = batch["features"]
        labels = batch["label"]
        _lipid_info = batch["info"]
        dataset_path = batch["dataset_path"]

        if isinstance(self.model, TransformerNetwork):
            labels_temp = labels[:, 1:]

            probabilities, tokens = self.model.predict_top_3(features)
            top_probs = torch.exp(probabilities[:, 0])
            top_probs = torch.unsqueeze(top_probs, dim=1)
            top_tokens = tokens[:, 0, 1:]

            assert self.test_custom_accuracy is not None
            accuracy_dict = self.test_custom_accuracy(top_tokens, labels_temp, "transformer", True)

            if (logger := self._get_custom_logger()) is not None:
                preds_vs_labels = self.get_test_preds_vs_labels(
                    top_tokens, labels_temp, dataset_path, top_probs
                )
                assert self.test_predictions is not None
                self.test_predictions(preds_vs_labels)
                logger.log_predictions(self.test_predictions, batch_idx, "test")

            loss = nn.functional.mse_loss(
                top_tokens.to(dtype=torch.float32), labels_temp.to(dtype=torch.float32)
            )

            self.log(
                "test_loss", loss.item(), on_step=True, sync_dist=True, batch_size=self.batch_size
            )
            self.log_dict(
                accuracy_dict,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
                batch_size=self.batch_size,
            )

        else:
            output = self.model(features)
            loss = nn.functional.mse_loss(output, labels)

        return loss

    def predict_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        features = batch["features"]
        spectrum_info = batch["info"]

        if isinstance(self.model, TransformerNetwork):
            probabilities, tokens = self.model.predict_top_3(features)

            if (logger := self._get_custom_logger()) is not None:
                logger.log_predictions(tokens, probabilities, spectrum_info)

    def get_preds_vs_labels(
        self, batch_idx: int, output: torch.Tensor, labels: torch.Tensor, dataset_path: torch.Tensor
    ) -> torch.Tensor:
        epoch_batch = torch.tensor([self.current_epoch, batch_idx]).type_as(output)
        epoch_batch = epoch_batch.repeat(len(output), 1)
        preds_vs_labels = torch.cat([epoch_batch, output, labels, dataset_path], dim=1)
        return preds_vs_labels

    def get_test_preds_vs_labels(
        self,
        output: torch.Tensor,
        labels: torch.Tensor,
        dataset_path: torch.Tensor,
        confidence_scores: torch.Tensor,
    ) -> torch.Tensor:
        preds_vs_labels = torch.cat([output, labels, dataset_path, confidence_scores], dim=1)
        return preds_vs_labels

    def get_neural_network(self) -> nn.Module:
        model_type = self.config.get("model")
        if model_type == "convolutional":
            return ConvolutionalNetwork(self.config)
        elif model_type == "transformer":
            return TransformerNetwork(self.config)
        elif model_type == "lstm":
            return LSTMNetwork(self.config)
        elif model_type == "feedforward":
            return FeedForwardNetwork(self.config)
        else:
            raise ValueError(f"Unknown model type: {model_type!r}")

    def save_model(self, output_folder: str) -> None:
        output_file = os.path.join(output_folder, "lipidetective_model.pth")
        torch.save(self.model.state_dict(), output_file)
        logging.info(f"Model saved to: {output_file}")

        metadata = extract_model_metadata(self.config)
        metadata_file = os.path.join(output_folder, "model_config.yaml")
        if write_yaml(metadata_file, metadata):
            logging.info(f"Model metadata saved to: {metadata_file}")
        else:
            logging.warning(f"Failed to save model metadata to: {metadata_file}")
