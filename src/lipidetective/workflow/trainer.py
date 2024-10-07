import torch
import h5py
import pytorch_lightning as pl
import os
import pandas as pd
import copy
import logging
import ray
import wandb

from random import shuffle
from datetime import datetime
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import Callback
from pytorch_lightning import loggers as pl_loggers
from ray import tune
from ray.tune.experiment import Trial
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from ray.tune.logger import LoggerCallback
from ray.air.integrations.wandb import WandbLoggerCallback

from src.lipidetective.helpers.utils import read_yaml, write_yaml, is_main_process, is_lipid_class_with_slash
from src.lipidetective.helpers.logging import Evaluator, CustomLogger, PredictionLogger
from src.lipidetective.models.random_forest import RandomForest
from src.lipidetective.workflow.h5_dataset import H5Dataset
from src.lipidetective.workflow.prediction_dataset import PredictionDataset
from src.lipidetective.helpers.lipid_library import LipidLibrary
from src.lipidetective.helpers.visualizations import generate_plots
from src.lipidetective.workflow.lightning_module import LightningModule


class Trainer:
    """The trainer class creates the lightning module and executes the specified workflow. It handles processing and
    splitting of the dataset.
    """

    def __init__(self, config):
        self.config = config

        if self.config['model'] != 'random_forest':
            self.log_every_n_steps = self.config['workflow']['log_every_n_steps']

            if is_main_process():
                self.time_initialized = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                self.output_folder = os.path.join(config['files']['output'],
                                                  f'LipiDetective_Output_{self.time_initialized}')

                os.mkdir(self.output_folder)
                logging.info(f"Output folder has been initialized: {self.output_folder}")

                # Writes the config file into the output folder so one can look at the exact parameters for each run
                write_yaml(os.path.join(self.output_folder, "config.yaml"), self.config)

            self.lipid_librarian = LipidLibrary()
            self.evaluator = Evaluator(self.lipid_librarian)

            if torch.cuda.is_available():
                # This sets a list of GPU names so one can choose exactly which GPUs to use, otherwise it uses all
                if self.config['cuda']['gpu_nr']:
                    self.devices = self.config['cuda']['gpu_nr']
                    logging.info(f"Cuda has been found, running with the following devices: {self.devices}")
                else:
                    self.devices = 'auto'
            else:
                self.devices = 'auto'

            self.nr_workers = self.config['training']['nr_workers'] if self.config['training']['nr_workers'] else 0

    def train_with_validation(self):
        trainsets, valsets, trainset_lipids, valset_lipids = self.perform_data_split()

        logging.info(f"Trainset: {len(trainsets[0])}")
        logging.info(f"Valset: {len(valsets[0])}")

        for fold in range(len(trainsets)):
            pl_module = LightningModule(self.config, self.evaluator, trainset_lipids=trainset_lipids[fold],
                                        valset_lipids=valset_lipids[fold])

            if len(trainsets) == 1:
                version = "."
                if 'wandb' in self.config:
                    wandb_name = self.time_initialized
                    group = self.config['wandb']['group'] if self.config['wandb']['group'] else None
            else:
                version = f"fold_{fold + 1}"
                if 'wandb' in self.config:
                    group = self.time_initialized
                    wandb_name = version

            train_loader = DataLoader(trainsets[fold], batch_size=self.config['training']['batch'],
                                      num_workers=self.nr_workers, shuffle=True)
            val_loader = DataLoader(valsets[fold], batch_size=self.config['training']['batch'],
                                    num_workers=self.nr_workers)

            tb_logger = pl_loggers.TensorBoardLogger(save_dir=self.output_folder,
                                                     name="tb_logger",
                                                     version=version)
            csv_logger = pl_loggers.CSVLogger(save_dir=self.output_folder,
                                              name="csv_logger",
                                              version=version)

            custom_logger = CustomLogger(save_dir=self.output_folder, version=version,
                                         log_every_n_steps=self.log_every_n_steps,
                                         config=self.config,
                                         do_training=True, do_validation=True,
                                         trainset_names=copy.deepcopy(trainsets[fold].dataset_names),
                                         valset_names=copy.deepcopy(valsets[fold].dataset_names),
                                         trainset_lipids=trainset_lipids, valset_lipids=valset_lipids)

            if 'wandb' in self.config:
                wandb.init(project="lipidetective", config=self.config, group=group, name=wandb_name)

            trainer = pl.Trainer(max_epochs=self.config['training']['epochs'], callbacks=PrintingCallbacks(),
                                 devices=self.devices, default_root_dir=self.output_folder,
                                 logger=[custom_logger, tb_logger, csv_logger],
                                 log_every_n_steps=self.log_every_n_steps, num_nodes=1)

            trainer.fit(model=pl_module, train_dataloaders=train_loader, val_dataloaders=val_loader)

            if self.config['model'] == 'transformer':
                custom_logger.save_lipid_wise_metrics(pl_module.train_custom_accuracy.metric.get_confusion_matrix(),
                                                      trainset_lipids[fold],
                                                      pl_module.val_custom_accuracy.metric.get_confusion_matrix(),
                                                      valset_lipids[fold])

            if self.config['workflow']['save_model']:
                if version != ".":
                    output_folder = os.path.join(self.output_folder, version)
                    pl_module.save_model(output_folder)
                else:
                    pl_module.save_model(self.output_folder)

    def train_without_validation(self):
        with h5py.File(self.config['files']['train_input'], 'r') as hdf5_file:
            dataset_names = list(hdf5_file['all_datasets'].keys())

        dataset = H5Dataset(self.config, dataset_names, self.lipid_librarian, self.config['files']['train_input'])
        dataset_lipids = self.get_unique_lipids(dataset_names)

        logging.info(f"Dataset selected for training contains {len(dataset)} spectra.")

        pl_module = LightningModule(self.config, self.evaluator, trainset_lipids=dataset_lipids)

        data_loader = DataLoader(dataset, batch_size=self.config['training']['batch'], num_workers=self.nr_workers,
                                 shuffle=True)

        custom_logger = CustomLogger(save_dir=self.output_folder, version=".", log_every_n_steps=self.log_every_n_steps,
                                     config=self.config, do_training=True,
                                     trainset_names=copy.deepcopy(dataset_names), trainset_lipids=dataset_lipids)

        if 'wandb' in self.config:
            group = self.config['wandb']['group'] if self.config['wandb']['group'] else None
            wandb.init(project="lipidetective", config=self.config, name=self.time_initialized, group=group)

        trainer = pl.Trainer(max_epochs=self.config['training']['epochs'], callbacks=PrintingCallbacks(),
                             logger=custom_logger, devices=self.devices, default_root_dir=self.output_folder,
                             log_every_n_steps=self.log_every_n_steps)

        trainer.fit(model=pl_module, train_dataloaders=data_loader)

        if self.config['model'] == 'transformer':
            custom_logger.save_lipid_wise_metrics(pl_module.train_custom_accuracy.metric.get_confusion_matrix(),
                                                  dataset_lipids)

        if self.config['workflow']['save_model']:
            pl_module.save_model(self.output_folder)

    def test(self):
        """
        This loop is for analyzing the models performance on a previously unseen labeled test dataset.
        """

        with h5py.File(self.config['files']['test_input'], 'r') as hdf5_file:
            dataset_names = list(hdf5_file['all_datasets'].keys())

        dataset = H5Dataset(self.config, dataset_names, self.lipid_librarian, self.config['files']['test_input'])
        dataset_lipids = self.get_unique_lipids(dataset_names)

        logging.info(f"Dataset selected for testing contains {len(dataset)} spectra.")

        pl_module = LightningModule(self.config, self.evaluator, testset_lipids=dataset_lipids)

        if self.config['test']['batch']:
            batch_size = self.config['test']['batch']
        else:
            batch_size = 1

        data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=self.nr_workers)

        custom_logger = CustomLogger(save_dir=self.output_folder, version=".", log_every_n_steps=1,
                                     config=self.config, do_testing=True,
                                     testset_names=copy.deepcopy(dataset_names), testset_lipids=dataset_lipids)

        trainer = pl.Trainer(callbacks=PrintingCallbacks(), logger=custom_logger, devices=1, deterministic=True,
                             default_root_dir=self.output_folder, log_every_n_steps=1)

        trainer.test(model=pl_module, dataloaders=data_loader)

        if self.config['model'] == 'transformer':
            custom_logger.save_lipid_wise_metrics(test_confusion_matrix=pl_module.test_custom_accuracy.metric.get_confusion_matrix(),
                                                  test_lipids=dataset_lipids)

    def predict(self):
        """
        Predicts lipid species for unlabeled data. Input is one or more mzML.
        """

        pl_module = LightningModule(self.config, self.evaluator)

        if self.config['predict']['batch']:
            batch_size = self.config['predict']['batch']
        else:
            batch_size = 1

        pred_logger = PredictionLogger(save_dir=self.output_folder, log_every_n_steps=1, config=self.config)

        trainer = pl.Trainer(callbacks=PrintingCallbacks(), logger=pred_logger, devices=1, deterministic=True, default_root_dir=self.output_folder, log_every_n_steps=1)

        pred_files = self.get_pred_files()

        for file in pred_files:
            dataset = PredictionDataset(file, self.config)
            logging.info(f"Dataset selected for prediction contains {len(dataset)} spectra.")
            data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=self.nr_workers)

            trainer.predict(model=pl_module, dataloaders=data_loader)

    def get_pred_files(self):
        input_path = self.config['files']['predict_input']

        if os.path.isfile(input_path):
            return [input_path]
        else:
            files = []
            for file in os.listdir(input_path):
                file_path = os.path.join(input_path, file)
                if os.path.isfile(file_path):
                    files.append(file_path)
            return files

    def schedule_tuning(self):
        logging.info(f"Scheduling tuning, cuda is available: {torch.cuda.is_available()}")
        trainsets, valsets, trainset_lipids, valset_lipids = self.perform_data_split()

        logging.info(f"Trainset: {len(trainsets[0])}")
        logging.info(f"Valset: {len(valsets[0])}")

        train_loader = DataLoader(trainsets[0], batch_size=self.config['training']['batch'],
                                  num_workers=self.nr_workers, shuffle=True)
        val_loader = DataLoader(valsets[0], batch_size=self.config['training']['batch'], num_workers=self.nr_workers)

        scheduler = ModifiedASHAScheduler(max_t=self.config['training']['epochs'],
                                          grace_period=self.config['tune']['grace_period'], reduction_factor=4)

        self.prepare_tune_config()

        if 'wandb' in self.config:
            callbacks = [CustomLoggerCallback(self.output_folder)]
            callbacks.append(WandbLoggerCallback(project="lipidetective"))
        else:
            callbacks = [CustomLoggerCallback(self.output_folder)]

        analysis = tune.run(
            tune.with_parameters(self.tune_model,
                                 num_epochs=self.config['training']['epochs'],
                                 train_loader=train_loader,
                                 val_loader=val_loader,
                                 trainset_lipids=trainset_lipids[0],
                                 valset_lipids=valset_lipids[0]
                                 ),
            config=self.config,
            fail_fast=True,
            metric="val_loss_epoch",
            mode="min",
            local_dir=self.output_folder,
            log_to_file=True,
            num_samples=self.config['tune']['nr_trials'],
            scheduler=scheduler,
            name="tune_results",
            callbacks=callbacks,
            resources_per_trial=self.config['tune']['resources_per_trial']
        )

        logging.info(f"Best hyperparameters found were: {analysis.best_config}")

        tune_result = os.path.join(self.output_folder, "tune_result.txt")
        with open(tune_result, "w") as file:
            string_to_write = f"Best hyperparameters found were: {analysis.best_config}"
            file.write(string_to_write)

    def tune_model(self, config, num_epochs, train_loader, val_loader, trainset_lipids, valset_lipids):
        pl_module = LightningModule(config, evaluator=self.evaluator, trainset_lipids=trainset_lipids,
                                    valset_lipids=valset_lipids)

        trial_dir = ray.train.get_context().get_trial_dir()

        tb_logger = pl_loggers.TensorBoardLogger(save_dir=trial_dir, name="tb_logger", version='.')
        csv_logger = pl_loggers.CSVLogger(save_dir=trial_dir, name="csv_logger", version='.')

        val_metrics = {"val_loss_epoch": "val_loss_epoch",
                   'val_accuracy_epoch': "customaccuracy_val_accuracy_epoch" }

        trainer = pl.Trainer(max_epochs=num_epochs,
                             devices=self.devices, default_root_dir=self.output_folder,
                             callbacks=[TuneReportCheckpointCallback(val_metrics, on="validation_end")],
                             logger=[tb_logger, csv_logger], log_every_n_steps=self.log_every_n_steps)

        trainer.fit(model=pl_module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    def prepare_tune_config(self):
        self.check_parameter_for_tuning(config_section='training', parameter='learning_rate')
        self.check_parameter_for_tuning(config_section='training', parameter='lr_step')
        self.check_parameter_for_tuning(config_section='training', parameter='batch')

        if self.config['model'] == 'transformer':
            self.check_parameter_for_tuning(config_section='transformer', parameter='d_model')
            self.check_parameter_for_tuning(config_section='transformer', parameter='num_heads')
            self.check_parameter_for_tuning(config_section='transformer', parameter='dropout')
            self.check_parameter_for_tuning(config_section='transformer', parameter='ffn_hidden')
            self.check_parameter_for_tuning(config_section='transformer', parameter='num_layers')

    def check_parameter_for_tuning(self, config_section: str, parameter: str):
        if type(self.config[config_section][parameter]) is list:
            self.config[config_section][parameter] = tune.grid_search(self.config[config_section][parameter])

    def parse_lipid_dataset_name(self, lipid_name):
        lipid_name = lipid_name.split(' | ')[0]
        if is_lipid_class_with_slash(lipid_name):
            lipid_name = lipid_name.replace('_', '/')
        return lipid_name

    def get_unique_lipids(self, dataset_list):
        dataset_lipids = set([self.parse_lipid_dataset_name(name) for name in dataset_list])
        dataset_lipids = list(dataset_lipids)
        return dataset_lipids

    def perform_data_split(self):
        """This method extracts the names of all datasets in the HDF5 input file and saves them in separate lists for the
        training and validation sets. These lists can than be used to iterate over the dataset using lazy loading if the
        whole dataset is too big to be loaded at once."""

        if self.config['files']['val_input']:
            with h5py.File(self.config['files']['train_input'], 'r') as hdf5_file:
                trainset_names = list(hdf5_file['all_datasets'].keys())
                shuffle(trainset_names)
                trainsets = [
                    H5Dataset(self.config, trainset_names, self.lipid_librarian, self.config['files']['train_input'])]

            with h5py.File(self.config['files']['val_input'], 'r') as hdf5_file:
                valset_names = list(hdf5_file['all_datasets'].keys())
                shuffle(valset_names)
                valsets = [
                    H5Dataset(self.config, valset_names, self.lipid_librarian, self.config['files']['val_input'])]

            trainset_lipids = [self.get_unique_lipids(
                trainset_names)]  # wrapping it in list so it works independent on if the data is split by fold or not
            valset_lipids = [self.get_unique_lipids(valset_names)]

        else:
            # Get all dataset names
            with h5py.File(self.config['files']['train_input'], 'r') as hdf5_file:
                all_dataset_names = list(hdf5_file['all_datasets'].keys())

            # Sort dataset names into train and validation set
            if self.config['files']['splitting_instructions']:
                # Split via instructions in YAML file
                trainsets, valsets, trainset_lipids, valset_lipids = self.split_data_via_instructions(all_dataset_names)
            else:
                # Split via lipid species into folds for cross-validation
                trainsets, valsets, trainset_lipids, valset_lipids = self.split_data_by_lipid_species(all_dataset_names)

        return trainsets, valsets, trainset_lipids, valset_lipids

    def split_data_via_instructions(self, dataset_names):
        trainset_names = []
        valset_names = []

        splitting_instructions = read_yaml(self.config['files']['splitting_instructions'])
        val_names = tuple(splitting_instructions['val'])

        for dataset_name in dataset_names:
            dataset_name_temp = dataset_name.split(' | ')[0]
            if dataset_name_temp in val_names:
                valset_names.append(dataset_name)
            else:
                trainset_names.append(dataset_name)

        shuffle(trainset_names)
        shuffle(valset_names)

        # Create HDF5Dataset objects for the train and validation set
        trainset = H5Dataset(self.config, trainset_names, self.lipid_librarian, self.config['files']['train_input'])
        valset = H5Dataset(self.config, valset_names, self.lipid_librarian, self.config['files']['train_input'])

        trainset_lipids = self.get_unique_lipids(trainset_names)
        valset_lipids = self.get_unique_lipids(valset_names)

        return [trainset], [valset], [trainset_lipids], [valset_lipids]

    def split_data_by_lipid_species(self, dataset_names):
        """Sorts the data by lipid type, mode and collision energy so the training and validation sets generated
        during the splitting can be balanced.
         """
        possible_lipid_species = list(set([name.split(' | ')[0] for name in dataset_names]))
        nr_folds = self.config['training']['k']

        number_lipids_excluded = int(len(possible_lipid_species) / nr_folds)

        trainset_names = []
        valset_names = []

        fold_lipids_excluded = []

        for i in range(1, nr_folds + 1):
            trainset_names.append([])
            valset_names.append([])

            index_end = i * number_lipids_excluded
            index_start = index_end - number_lipids_excluded
            lipids_excluded_in_fold = possible_lipid_species[index_start:index_end]

            fold_lipids_excluded.append(tuple(lipids_excluded_in_fold))

        for lipid_name in dataset_names:
            for i in range(nr_folds):
                if lipid_name.startswith(fold_lipids_excluded[i]):
                    valset_names[i].append(lipid_name)
                else:
                    trainset_names[i].append(lipid_name)

        trainsets = []
        valsets = []

        trainsets_lipids = []
        valsets_lipids = []

        for fold in trainset_names:
            dataset = H5Dataset(self.config, fold, self.lipid_librarian, self.config['files']['train_input'])
            trainsets.append(dataset)
            trainset_lipids = self.get_unique_lipids(fold)
            trainsets_lipids.append(trainset_lipids)

        for fold in valset_names:
            dataset = H5Dataset(self.config, fold, self.lipid_librarian, self.config['files']['train_input'])
            valsets.append(dataset)
            valset_lipids = self.get_unique_lipids(fold)
            valsets_lipids.append(valset_lipids)

        return trainsets, valsets, trainsets_lipids, valsets_lipids

    def run_random_forest(self):
        random_forest = RandomForest(self.config)
        random_forest.run()

        logging.info('Random forest run completed.')


class ModifiedASHAScheduler(ASHAScheduler):
    def on_trial_add(self, trial_runner, trial: Trial):
        trial_id = int(str(trial).split('_')[-1])
        trial.config['trial_id'] = trial_id
        super().on_trial_add(trial_runner, trial)


class PrintingCallbacks(Callback):
    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print(f"\tLearning rate: {pl_module.lr_schedulers().get_last_lr()[0]}")


class CustomLoggerCallback(LoggerCallback):
    """Custom logger interface"""

    def __init__(self, output_folder):
        self.output_folder = output_folder

    def on_trial_complete(self, iteration, trials, trial, **info):
        print(f"Trial {trial} successfully completed.")
        csv_metrics_file = f"{trial.logdir}/csv_logger/metrics.csv"
        metrics = pd.read_csv(csv_metrics_file)
        trial_id = int(trial.trial_id.split('_')[-1])
        generate_plots(metrics, trial_id, self.output_folder, trial.evaluated_params)
