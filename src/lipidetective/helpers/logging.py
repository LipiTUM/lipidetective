import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch

from typing import Dict, Optional
from matplotlib.ticker import ScalarFormatter
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.utilities import rank_zero_only
from torchmetrics import Metric

from src.lipidetective.helpers.lipid_library import LipidLibrary


class Evaluator:
    def __init__(self, library):
        self.library: LipidLibrary = library

    def evaluate_regression_accuracy(self, predictions, lipid_info):
        nr_correct = 0

        for idx, prediction in enumerate(predictions):
            # Get predicted and label headgroups and check if they are the same
            label_hg = lipid_info['headgroup'][idx]
            prediction = prediction.squeeze()
            pred_hg, pred_hg_value = self.find_nearest_headgroup(prediction[0])
            hg_correct = label_hg == pred_hg

            # Same for both side chains
            label_sc1 = lipid_info['fatty_acid_sn1'][idx]
            pred_sc1, pred_sc1_value = self.find_nearest_side_chain(prediction[1])
            sc1_correct = label_sc1 == pred_sc1

            label_sc2 = lipid_info['fatty_acid_sn2'][idx]
            pred_sc2, pred_sc2_value = self.find_nearest_side_chain(prediction[2])
            sc2_correct = label_sc2 == pred_sc2

            # Check if all lipid components were predicted correctly
            if all([hg_correct, sc1_correct, sc2_correct]):
                nr_correct += 1

        return nr_correct, len(predictions)

    def evaluate_custom_transformer_accuracy(self, predictions, labels, lipid_name_dict, is_last_epoch):
        accuracy_sum = 0
        nr_correct = 0
        nr_lipids = len(lipid_name_dict)
        prediction_matrix = torch.zeros((nr_lipids + 1, nr_lipids))

        for idx, prediction in enumerate(predictions):
            prediction = ''.join(self.library.translate_tokens_to_name(prediction))
            label = ''.join(self.library.translate_tokens_to_name(labels[idx]))
            custom_accuracy = self.library.custom_accuracy_scoring(prediction, label)
            if custom_accuracy == 1:
                nr_correct += 1
            accuracy_sum += custom_accuracy

            if is_last_epoch:
                prediction = prediction.split(' [M')[0]
                label = label.split(' [M')[0]

                if label in lipid_name_dict.keys():  # skip for noise/blank spectra
                    if prediction in lipid_name_dict:
                        prediction_index = lipid_name_dict[prediction]
                        label_index = lipid_name_dict[label]
                        prediction_matrix[prediction_index, label_index] += 1
                    else:
                        label_index = lipid_name_dict[label]
                        prediction_matrix[nr_lipids, label_index] += 1
                else:
                    if prediction == label:
                        noise_index = lipid_name_dict['noise_spectrum']
                        prediction_matrix[noise_index, noise_index] += 1
                    elif prediction in lipid_name_dict:
                        prediction_index = lipid_name_dict[prediction]
                        noise_index = lipid_name_dict['noise_spectrum']
                        prediction_matrix[prediction_index, noise_index] += 1
                    else:
                        noise_index = lipid_name_dict['noise_spectrum']
                        prediction_matrix[nr_lipids, noise_index] += 1

        return accuracy_sum, nr_correct, len(predictions), prediction_matrix

    def generate_prediction_info(self, label_hg, label_sc1, label_sc2, pred_hg, pred_sc1, pred_sc2, pred_hg_value,
                                 pred_sc1_value, pred_sc2_value, batch, epoch, idx):
        label_hg_value = self.library.headgroups_mass_norm[label_hg]
        label_sc1_value = self.library.side_chains_mass_norm[label_sc1]
        label_sc2_value = self.library.side_chains_mass_norm[label_sc2]

        prediction_info = {
            'epoch': epoch,
            'batch': batch,
            'idx': idx,
            'pred_hg': pred_hg,
            'label_hg': label_hg,
            'pred_sc1': pred_sc1,
            'label_sc1': label_sc1,
            'pred_sc2': pred_sc2,
            'label_sc2': label_sc2,
            'pred_hg_value': pred_hg_value,
            'label_hg_value': label_hg_value,
            'pred_sc1_value': pred_sc1_value,
            'label_sc1_value': label_sc1_value,
            'pred_sc2_value': pred_sc2_value,
            'label_sc2_value': label_sc2_value,
        }

        return prediction_info

    def find_nearest_headgroup(self, value):
        hg_key, hg_val = min(self.library.headgroups_mass_norm.items(), key=lambda x: abs(value - x[1]))

        return hg_key, hg_val

    def find_nearest_side_chain(self, value):
        sc_key, sc_val = min(self.library.side_chains_mass_norm.items(), key=lambda x: abs(value - x[1]))

        return sc_key, sc_val


class CustomAccuracy(Metric):
    def __init__(self, evaluator, lipid_species_names):
        super().__init__()

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("accuracy_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

        self.lipid_name_dict = {k: v for v, k in enumerate(lipid_species_names)}
        nr_lipid_species = len(self.lipid_name_dict)

        self.add_state("confusion_matrix", default=torch.zeros((nr_lipid_species + 1, nr_lipid_species)), dist_reduce_fx='sum')

        self.evaluator = evaluator

    def update(self, predictions, lipid_info, model: str, is_last_epoch: bool = False):
        if model == "transformer":
            accuracy_sum, nr_correct, total, prediction_matrix = self.evaluator.evaluate_custom_transformer_accuracy(
                predictions, lipid_info, self.lipid_name_dict, is_last_epoch)
            self.accuracy_sum += accuracy_sum

            if is_last_epoch and self._to_sync:
                self.confusion_matrix += prediction_matrix.type_as(self.confusion_matrix)
        else:
            nr_correct, total = self.evaluator.evaluate_regression_accuracy(predictions, lipid_info)

        self.correct += nr_correct
        self.total += total

    def compute(self):
        accuracy = self.correct / self.total
        mean_accuracy = self.accuracy_sum.float() / self.total
        return torch.tensor([accuracy, mean_accuracy])

    def get_confusion_matrix(self):
        return self.confusion_matrix


class CustomLogger(Logger):
    def __init__(self, save_dir, version, log_every_n_steps, config, trainset_names=None,
                 valset_names=None, testset_names=None, do_training=False, do_validation=False, do_testing=False, trainset_lipids=None,
                 valset_lipids=None, testset_lipids=None):
        super().__init__()
        self.model = config['model']
        if self.model == 'transformer':
            self.output_seq_length = config['transformer']['output_seq_length'] - 1  # -1 because we don't save the <SOS> token

        self.librarian = LipidLibrary()
        self.output_directory = save_dir
        self.fold = version
        self.log_every_n_steps = log_every_n_steps
        self.do_training = do_training
        self.do_validation = do_validation
        self.do_testing = do_testing

        self.trainset_names = trainset_names
        self.valset_names = valset_names
        self.testset_names = testset_names
        self.trainset_lipids = trainset_lipids
        self.valset_lipids = valset_lipids
        self.testset_lipids = testset_lipids

        path_logger = os.path.join(self.output_directory, self.name)

        if not os.path.exists(path_logger):
            os.mkdir(path_logger)

        if self.fold != ".":
            self.save_path = os.path.join(path_logger, self.fold)
            os.mkdir(self.save_path)
        else:
            self.save_path = path_logger

        if do_training:
            self.train_csv_path, self.train_predictions_path = self.generate_output_files("train")

        if do_validation:
            self.val_csv_path, self.val_predictions_path = self.generate_output_files("validation")

        if do_testing:
            self.test_csv_path, self.test_predictions_path = self.generate_output_files("test")

    def generate_output_files(self, mode):
        csv_path = os.path.join(self.save_path, f"{mode}_metrics.csv")
        predictions_path = os.path.join(self.save_path, f"{mode}_predictions.csv")

        if self.model == 'transformer':
            with open(csv_path, 'w') as f:
                writer = csv.writer(f)
                if mode == 'test':
                    writer.writerow(["step", "loss", "accuracy", "average_accuracy"])
                else:
                    writer.writerow(["epoch", "loss", "accuracy", "average_accuracy"])

            with open(predictions_path, 'w') as f:
                writer = csv.writer(f)
                if mode == 'test':
                    writer.writerow(["pred", "label", "dataset_path", "average_accuracy", "custom_correct", "correct"])
                else:
                    writer.writerow(["epoch", "batch", "pred", "label", "dataset_path", "average_accuracy",
                                     "custom_correct", "correct"])
        else:
            with open(csv_path, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "loss", "accuracy", "R2", "mae_hg", "mae_fa1", "mae_fa2"])

            with open(predictions_path, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["epoch", "batch", "hg_pred", "sc1_pred", "sc2_pred", "hg_label", "sc1_label", "sc2_label"])

        return csv_path, predictions_path

    @property
    def name(self):
        return "custom_logger"

    @property
    def experiment(self):
        return None

    @property
    def save_dir(self):
        return self.output_directory

    @property
    def version(self):
        # Return the experiment version, int or str.
        return self.fold

    @rank_zero_only
    def log_hyperparams(self, params):
        # params is an argparse.Namespace
        # code to record hyperparameters goes here
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        # metrics is a dictionary of metric names and values
        # code to record metrics goes here

        if "train_loss_epoch" in metrics:
            with open(self.train_csv_path, 'a') as f:
                writer = csv.writer(f)

                if self.model == 'transformer':
                    writer.writerow([metrics['epoch'], metrics['train_loss_epoch'],
                                     "{:.2f}".format(metrics['customaccuracy_train_accuracy_epoch'] * 100),
                                     "{:.2f}".format(metrics['customaccuracy_train_mean_accuracy_epoch'] * 100)])
                else:
                    r2_score = metrics['train_r2_epoch'] if "train_r2_epoch" in metrics else 0
                    writer.writerow(
                        [metrics['epoch'], metrics['train_loss_epoch'], metrics['customaccuracy_train_accuracy_epoch'], r2_score,
                         metrics['train_mae_hg_epoch'], metrics['train_mae_fa1_epoch'], metrics['train_mae_fa2_epoch']])

        elif "val_loss_epoch" in metrics:
            with open(self.val_csv_path, 'a') as f:
                writer = csv.writer(f)

                if self.model == 'transformer':
                    writer.writerow([metrics['epoch'], metrics['val_loss_epoch'],
                                     "{:.2f}".format(metrics['customaccuracy_val_accuracy_epoch'] * 100),
                                     "{:.2f}".format(metrics['customaccuracy_val_mean_accuracy_epoch'] * 100)])
                else:
                    r2_score = metrics['val_r2_epoch'] if "val_r2_epoch" in metrics else 0
                    writer.writerow(
                        [metrics['epoch'], metrics['val_loss_epoch'], metrics['customaccuracy_val_accuracy_epoch'], r2_score,
                         metrics['val_mae_hg_epoch'], metrics['val_mae_fa1_epoch'], metrics['val_mae_fa2_epoch']])

        elif "test_loss_step" in metrics:
            with open(self.test_csv_path, 'a') as f:
                writer = csv.writer(f)

                if self.model == 'transformer':
                    writer.writerow([step, metrics['test_loss_step'], "{:.2f}".format(metrics['customaccuracy_test_accuracy_step'] * 100),
                                     "{:.2f}".format(metrics['customaccuracy_test_mean_accuracy_step'] * 100)])

    @rank_zero_only
    def log_predictions(self, cat_metric, batch_idx, workflow):
        if workflow == "train":
            if batch_idx % self.log_every_n_steps == 0:
                preds_vs_labels = cat_metric.compute().detach().cpu().numpy()

                if self.model == 'transformer':
                    preds_vs_labels = self.transform_token_predictions_to_string(preds_vs_labels, self.trainset_names)

                with open(self.train_predictions_path, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerows(preds_vs_labels)

        elif workflow == "val":
            preds_vs_labels = cat_metric.compute().detach().cpu().numpy()

            if self.model == 'transformer':
                preds_vs_labels = self.transform_token_predictions_to_string(preds_vs_labels, self.valset_names)

            with open(self.val_predictions_path, 'a') as f:
                writer = csv.writer(f)
                writer.writerows(preds_vs_labels)

        elif workflow == "test":
            preds_vs_labels = cat_metric.compute().detach().cpu().numpy()

            if self.model == 'transformer':
                preds_vs_labels = self.transform_test_token_predictions_to_string(preds_vs_labels, self.testset_names)

            with open(self.test_predictions_path, 'a') as f:
                writer = csv.writer(f)
                writer.writerows(preds_vs_labels)

        cat_metric.reset()

    def transform_token_predictions_to_string(self, preds_vs_labels, dataset_names):
        translated_tokens = []

        for row in preds_vs_labels:
            epoch = row[0]
            batch = row[1]
            prediction = ''.join(self.librarian.translate_tokens_to_name(row[2:self.output_seq_length + 2])).replace(
                '<PAD>', '')
            label = ''.join(self.librarian.translate_tokens_to_name(
                row[self.output_seq_length + 2:(2 * self.output_seq_length) + 2])).replace('<PAD>', '')
            dataset_path = dataset_names[int(row[(2 * self.output_seq_length) + 2])]
            correct = prediction == label
            custom_accuracy = self.librarian.custom_accuracy_scoring(prediction, label)
            custom_correct = custom_accuracy == 1
            translated_tokens.append(
                [epoch, batch, prediction, label, dataset_path, custom_accuracy, custom_correct, correct])

        translated_tokens = np.array(translated_tokens)
        return translated_tokens

    def transform_test_token_predictions_to_string(self, preds_vs_labels, dataset_names):
        translated_tokens = []

        for row in preds_vs_labels:
            prediction = ''.join(self.librarian.translate_tokens_to_name(row[0:self.output_seq_length])).replace(
                '<PAD>', '')
            label = ''.join(self.librarian.translate_tokens_to_name(
                row[self.output_seq_length:(2 * self.output_seq_length)])).replace('<PAD>', '')
            confidence_score = row[-1]
            dataset_path = dataset_names[int(row[(2 * self.output_seq_length)])]
            correct = prediction == label
            custom_accuracy = self.librarian.custom_accuracy_scoring(prediction, label)
            custom_correct = custom_accuracy == 1
            translated_tokens.append(
                [prediction, label, dataset_path, custom_accuracy, custom_correct, correct, confidence_score])

        translated_tokens = np.array(translated_tokens)
        return translated_tokens

    @rank_zero_only
    def save(self):
        # Optional. Any code necessary to save logger data goes here
        pass

    @rank_zero_only
    def finalize(self, status):
        # Optional. Any code that needs to be run after training finishes goes here
        if self.do_training:
            train_csv = pd.read_csv(self.train_csv_path)
            self.plot_loss_and_accuracy(train_csv, "training", self.save_path)

            if not self.model == 'transformer':
                self.plot_loss_and_mae(train_csv, "training", self.save_path)
                self.plot_loss_and_r2(train_csv, "training", self.save_path)
            else:
                self.plot_loss_and_both_accuracies(train_csv, "training", self.save_path)

        if self.do_validation:
            val_csv = pd.read_csv(self.val_csv_path)
            self.plot_loss_and_accuracy(val_csv, "validation", self.save_path)

            if not self.model == 'transformer':
                self.plot_loss_and_mae(val_csv, "validation", self.save_path)
                self.plot_loss_and_r2(val_csv, "validation", self.save_path)
            else:
                self.plot_loss_and_both_accuracies(val_csv, "validation", self.save_path)

    def save_lipid_wise_metrics(self, train_confusion_matrix=None, train_lipids=None, val_confusion_matrix=None, val_lipids=None, test_confusion_matrix=None, test_lipids=None):
        if train_confusion_matrix is not None:
            confusion_matrix_train_df = pd.DataFrame(train_confusion_matrix, columns=train_lipids,
                                                     index=train_lipids + ['Other'])
            confusion_matrix_train_df.to_csv(os.path.join(self.save_path, f"confusion_matrix_train.csv"))

            train_metrics_df = self.calculate_lipid_metrics(confusion_matrix_train_df)
            train_metrics_df.to_csv(os.path.join(self.save_path, f"train_lipid_metrics.csv"))

            if len(train_metrics_df) < 25:
                self.plot_confusion_matrix_as_heatmap(confusion_matrix_train_df, 'train')

        if val_confusion_matrix is not None:
            confusion_matrix_val_df = pd.DataFrame(val_confusion_matrix, columns=val_lipids,
                                                   index=val_lipids + ['Other'])
            confusion_matrix_val_df.to_csv(os.path.join(self.save_path, f"confusion_matrix_val.csv"))

            val_metrics_df = self.calculate_lipid_metrics(confusion_matrix_val_df)
            val_metrics_df.to_csv(os.path.join(self.save_path, f"val_lipid_metrics.csv"))

            self.plot_confusion_matrix_as_heatmap(confusion_matrix_val_df, 'validation')

        if test_confusion_matrix is not None:
            confusion_matrix_test_df = pd.DataFrame(test_confusion_matrix, columns=test_lipids,
                                                   index=test_lipids + ['Other'])
            confusion_matrix_test_df.to_csv(os.path.join(self.save_path, f"confusion_matrix_test.csv"))

            test_metrics_df = self.calculate_lipid_metrics(confusion_matrix_test_df)
            test_metrics_df.to_csv(os.path.join(self.save_path, f"test_lipid_metrics.csv"))

            self.plot_confusion_matrix_as_heatmap(confusion_matrix_test_df, 'testing')

    def calculate_lipid_metrics(self, confusion_matrix):
        confusion_matrix['sum_predictions'] = confusion_matrix.sum(axis=1)

        calculated_scores = []

        for lipid in confusion_matrix.columns[:-1]:
            correct = confusion_matrix.loc[lipid, lipid]
            total_number_predictions = confusion_matrix.loc[lipid, 'sum_predictions']

            if total_number_predictions != 0:
                precision = correct / total_number_predictions
            else:
                precision = 0

            total_number_label = confusion_matrix.loc[:, lipid].sum()
            if total_number_label != 0:
                recall = correct / total_number_label
            else:
                recall = 0

            if precision != 0 or recall != 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0

            calculated_scores.append({'lipid': lipid, 'precision': precision, 'recall': recall, 'f1': f1})

        calculated_scores_df = pd.DataFrame(calculated_scores)
        calculated_scores_df.set_index('lipid', inplace=True)

        return calculated_scores_df

    def plot_loss_and_accuracy(self, df, workflow, output_folder):
        plot_name = f'plot_loss_accuracy_{workflow}_{self.fold}.png' if self.fold != "." else f'plot_loss_accuracy_{workflow}.png'
        plot_name_loss = f'plot_loss_{workflow}_{self.fold}.png' if self.fold != "." else f'plot_loss_{workflow}.png'

        df.index = df.epoch

        figure = plt.figure(figsize=(12, 10))

        ax1 = sns.lineplot(data=df.loss, color='orange')
        plt.ylabel('Loss', fontsize=14)
        ax1.set_yscale('log')
        plt.xlabel('Epoch', fontsize=14)
        ax1.legend(['Loss'], loc=(1.15, .92), frameon=False, fontsize=13)

        plt.savefig(os.path.join(output_folder, plot_name_loss), dpi=300)

        ax2 = plt.twinx()
        sns.lineplot(data=df.accuracy, color='b', ax=ax2)
        plt.ylabel('Accuracy in %', fontsize=14)
        y_ticks = list(range(0, 101, 10))
        ax2.set_yticks(y_ticks)
        ax2.legend(['Accuracy'], loc=(1.15, .95), frameon=False, fontsize=13)

        plt.title(f'Loss and Accuracy per Epoch', fontsize=16)
        plt.tight_layout(w_pad=4)
        plt.savefig(os.path.join(output_folder, plot_name), dpi=300)
        plt.close(figure)

    def plot_loss_and_both_accuracies(self, df, workflow, output_folder):
        plot_name = f'plot_loss_accuracies_{workflow}_{self.fold}.png' if self.fold != "." else f'plot_loss_accuracies_{workflow}.png'
        plot_name_loss_custom_accuracy = f'plot_loss_custom_accuracy_{workflow}_{self.fold}.png' if self.fold != "." else f'plot_loss_custom_accuracy_{workflow}.png'

        df.index = df.epoch

        figure = plt.figure(figsize=(12, 10))

        ax1 = sns.lineplot(data=df.loss, color='orange')
        plt.ylabel('Loss', fontsize=14)
        ax1.set_yscale('log')
        plt.xlabel('Epoch', fontsize=14)
        ax1.legend(['Loss'], loc=(1.1, .92), frameon=False, fontsize=13)

        ax2 = plt.twinx()
        sns.lineplot(data=df.average_accuracy, color='g', ax=ax2)
        plt.ylabel('Accuracy in %', fontsize=14)
        y_ticks = list(range(0, 101, 10))
        ax2.set_yticks(y_ticks)
        ax2.legend(['Mean Accuracy / Lipid'], loc=(1.1, .95), frameon=False, fontsize=13)

        plt.title(f'Loss and Accuracy per Epoch', fontsize=16)
        plt.tight_layout(w_pad=4)

        plt.savefig(os.path.join(output_folder, plot_name_loss_custom_accuracy), dpi=300)

        sns.lineplot(data=df.accuracy, color='b', ax=ax2)
        ax2.set_yticks(y_ticks)
        ax2.legend(['Mean Accuracy / Lipid', '_', 'Accuracy'], loc=(1.1, .95), frameon=False, fontsize=13)
        plt.tight_layout(w_pad=4)

        plt.savefig(os.path.join(output_folder, plot_name), dpi=300)
        plt.close(figure)

    def plot_loss_and_mae(self, df, workflow, output_folder):
        plot_name = f'plot_loss_mae_{workflow}_{self.fold}.png' if self.fold != "." else f'plot_loss_mae_{workflow}.png'
        plot_name_mae = f'plot_mae_{workflow}_{self.fold}.png' if self.fold != "." else f'plot_mae_{workflow}.png'

        df.index = df.epoch

        figure = plt.figure(figsize=(12, 10))

        ax1 = sns.lineplot(data=df.mae_hg, color='darkorchid')
        sns.lineplot(data=df.mae_fa1, color='cornflowerblue', ax=ax1)
        sns.lineplot(data=df.mae_fa2, color='chocolate', ax=ax1)
        ax1.legend(['mae_hg', '_', 'mae_fa1', '_', 'mae_fa2'], loc=(1.15, .95), frameon=False, fontsize=13)
        plt.ylabel('MAE', fontsize=14)
        ax1.set_yscale('log')

        plt.tight_layout(w_pad=4)
        plt.savefig(os.path.join(output_folder, plot_name_mae), dpi=300)

        ax2 = plt.twinx()
        sns.lineplot(data=df.loss, color='orange', ax=ax2)
        plt.ylabel('Loss', fontsize=14)
        ax2.set_yscale('log')
        plt.xlabel('Epoch', fontsize=14)
        ax2.legend(['Loss'], loc=(1.15, .92), frameon=False, fontsize=13)

        plt.title(f'Loss and MAE per Epoch', fontsize=16)
        plt.tight_layout(w_pad=4)
        plt.savefig(os.path.join(output_folder, plot_name), dpi=300)
        plt.close(figure)

    def plot_loss_and_r2(self, df, workflow, output_folder):
        plot_name = f'plot_loss_r2_{workflow}_{self.fold}.png' if self.fold != "." else f'plot_loss_r2_{workflow}.png'
        df.index = df.epoch

        figure = plt.figure(figsize=(12, 10))

        ax1 = sns.lineplot(data=df.loss, color='orange')
        plt.ylabel('Loss', fontsize=14)
        ax1.set_yscale('log')
        plt.xlabel('Epoch', fontsize=14)
        ax1.legend(['Loss'], loc=(1.15, .92), frameon=False, fontsize=13)

        ax2 = plt.twinx()
        sns.lineplot(data=df.R2, color='lightgreen', ax=ax2)
        plt.ylabel('R2', fontsize=14)
        ax2.set_yscale('symlog')
        ax2.yaxis.set_major_formatter(ScalarFormatter())
        ax2.set_ylim(top=3)
        ax2.legend(['R2'], loc=(1.15, .95), frameon=False, fontsize=13)

        plt.title(f'Loss and R2 per Epoch', fontsize=16)
        plt.tight_layout(w_pad=4)
        plt.savefig(os.path.join(output_folder, plot_name), dpi=300)
        plt.close(figure)

    def plot_confusion_matrix_as_heatmap(self, confusion_matrix, workflow):
        plot_name = f'confusion_matrix_heatmap_{workflow}.png'

        figure = plt.figure(figsize=(12, 10))

        plt.title(f'Confusion Matrix of Lipid Species {workflow} Predictions')
        sns.heatmap(confusion_matrix.drop('sum_predictions', axis=1), cmap="rocket_r", fmt='g', annot=True,
                    linewidth=.5, yticklabels=True, xticklabels=True)
        plt.tight_layout(w_pad=4)
        plt.savefig(os.path.join(self.save_path, plot_name), dpi=300)
        plt.close(figure)


class PredictionLogger(Logger):

    def __init__(self, save_dir, log_every_n_steps, config):
        super().__init__()
        self.model = config['model']
        if self.model == 'transformer':
            self.output_seq_length = config['transformer']['output_seq_length'] - 1  # -1 because we don't save the <SOS> token

        self.librarian = LipidLibrary()
        self.output_directory = save_dir
        self.log_every_n_steps = log_every_n_steps

        if 'predict' in config:
            self.save_top3 = config['predict']['output'] == 'top3' if 'output' in config['predict'] else False
            self.save_spectrum = config['predict']['save_spectrum'] if 'save_spectrum' in config['predict'] else False
            self.keep_empty = config['predict']['keep_empty'] if 'keep_empty' in config['predict'] else False
            self.keep_wrong_polarity_preds = config['predict']['keep_wrong_polarity_preds'] if 'keep_wrong_polarity_preds' in config['predict'] else False
            self.confidence_threshold = config['predict']['confidence_threshold'] if 'confidence_threshold' in config['predict'] else 0
        else:
            self.save_top3 = False
            self.save_spectrum = False
            self.keep_empty = False
            self.keep_wrong_polarity_preds = False
            self.confidence_threshold = 0

        self.predictions_path, self.top_3_predictions_path = self.generate_output_files()

    def generate_output_files(self):
        predictions_path = os.path.join(self.output_directory, f"predictions.csv")

        with open(predictions_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["file", "polarity", "spectrum_index", "precursor", "prediction", "confidence"])

        if self.save_top3:
            top_3_predictions_path = os.path.join(self.output_directory, f"top3_predictions.csv")
            with open(top_3_predictions_path, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(["file", "spectrum_index", "prediction_1", "confidence_1", "prediction_2", "confidence_2", "prediction_3", "confidence_3"])
        else:
            top_3_predictions_path = None

        return predictions_path, top_3_predictions_path

    @property
    def name(self):
        return "custom_logger"

    @property
    def experiment(self):
        return None

    @property
    def save_dir(self):
        return self.output_directory

    @property
    def version(self):
        # Return the experiment version, int or str.
        return '.'

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        pass

    @rank_zero_only
    def log_hyperparams(self, params):
        # params is an argparse.Namespace
        # code to record hyperparameters goes here
        pass

    @rank_zero_only
    def log_predictions(self, tokens, probabilities, spectrum_info):
        rows, rows_top3 = self.transform_test_token_predictions_to_string(tokens, probabilities, spectrum_info)

        with open(self.predictions_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerows(rows)

        if self.save_top3:
            with open(self.top_3_predictions_path, 'a') as f:
                writer = csv.writer(f)
                writer.writerows(rows_top3)

    def transform_test_token_predictions_to_string(self, tokens, probabilities, spectrum_info):
        translated_tokens = []
        translated_tokens_top_3 = []

        for i in range(len(tokens)):
            spectrum_index = spectrum_info['index'][i]
            if type(spectrum_index) is not str:
                spectrum_index = spectrum_index.item()
            file_name = spectrum_info['file'][i]
            polarity = spectrum_info['polarity'][i]
            precursor = spectrum_info['precursor'][i]
            spectrum_predictions = []

            for j in range(3):
                prob = torch.exp(probabilities[i, j])
                confidence_score = prob.item()
                tokens_temp = tokens[i, j, 1:]
                prediction = ''.join(self.librarian.translate_tokens_to_name(tokens_temp)).replace('<PAD>', '').replace('<EOS>', '')
                spectrum_predictions.append([prediction, confidence_score])

            translated_tokens_row = [file_name, polarity, spectrum_index, precursor]
            translated_tokens_row.extend(spectrum_predictions[0])

            polarity_prediction = spectrum_predictions[0][0][-1] if spectrum_predictions[0][0] else polarity

            if spectrum_predictions[0][1] > self.confidence_threshold:
                if polarity_prediction != polarity and not self.keep_wrong_polarity_preds:
                    spectrum_predictions[0][0] = ''
                if not spectrum_predictions[0][0] == '' or self.keep_empty:
                    translated_tokens.append(translated_tokens_row)

            translated_tokens_top_3_row = [file_name, spectrum_index]
            for entry in spectrum_predictions:
                translated_tokens_top_3_row.extend(entry)
            translated_tokens_top_3.append(translated_tokens_top_3_row)

        translated_tokens = np.array(translated_tokens)
        translated_tokens_top_3 = np.array(translated_tokens_top_3)

        return translated_tokens, translated_tokens_top_3

    @rank_zero_only
    def save(self):
        # Optional. Any code necessary to save logger data goes here
        pass

    @rank_zero_only
    def finalize(self, status):
        # Optional. Any code that needs to be run after training finishes goes here
        pass
