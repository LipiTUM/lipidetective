import torch
import h5py
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from src.lipidetective.helpers.utils import truncate
from src.lipidetective.helpers.lipid_library import LipidLibrary


class H5Dataset(Dataset):
    """
    This class implements a custom PyTorch dataset. It handles reading in the training data from HDF5 files. It
    overrides the __getitem__ method to return the processed spectrum and its metadata for a sample at a given index.

    Attributes:
        file_path (str): The path to the HDF5 file
        dataset_names (list): list of dataset names in the HDF5 file group "all_datasets", used to access samples by index
        dataset_len (int): An integer count of the samples in the HDF5 file
        hdf5_file (h5py.Dataset): the opened HDF5 file, set to None during initialization and set once first sample is requested
        config (dict): Dictionary containing the information from the config.yaml file
        network_type (str): String of the network type specified in the config.yaml file
        decimal_accuracy (int): Integer indicating the decimal accuracy to which the mass spectra should be binned
        lipid_librarian (LipidLibrary): LipidLibrarian instance used for generating the label for a sample
    """
    def __init__(self, config: dict, dataset_names: list, lipid_librarian: LipidLibrary, file_path: str):
        self.file_path = file_path
        self.dataset_names = dataset_names
        self.dataset_len = len(self.dataset_names)
        self.hdf5_file = None

        self.config = config
        self.network_type = config['model']
        self.decimal_accuracy = self.config['input_embedding']['decimal_accuracy']

        self.lipid_librarian = lipid_librarian

    def __len__(self) -> int:
        return self.dataset_len

    def __getitem__(self, index: int) -> dict:
        # Handles first opening of HDF5 file and ensures it is only opened once for all parallel processes
        if self.hdf5_file is None:
            self.hdf5_file = h5py.File(self.file_path, 'r')

        # Extracts spectrum with requested index from HDF5 dataset
        sample: h5py.Dataset = self.hdf5_file[f"/all_datasets/{self.dataset_names[index]}"]
        spectrum: np.ndarray = sample[()]
        features = self.get_n_highest_peaks(spectrum, self.config['input_embedding']['n_peaks'])

        if self.network_type == "transformer":
            # Extract only sorted m/z values as we don't need the intensities for the transformer input
            features = features[:, 0]
            features = torch.IntTensor(features * (10 ** self.decimal_accuracy))

            precursor_mz = int(float(sample.attrs['precursor']) * (10 ** self.decimal_accuracy))

            if precursor_mz not in features:
                features[-1] = precursor_mz

            # Scale float m/z values to ints, so they can function as an index to be immediately mapped to their
            # respective embedding vector

            # Create labels and save dataset path, so we can trace each prediction back to it's input spectrum
            sample_label, sample_info = self.lipid_librarian.get_transformer_label(
                sample.attrs['lipid_species'],
                sample.attrs['adduct'],
                self.config['transformer']['output_seq_length'])

            sample_info['dataset_name'] = self.dataset_names[index]

        else:
            features[:, 0] = features[:, 0] / self.config['input_embedding']['max_mz']
            features = torch.FloatTensor(features)
            polarity = 1 if sample.attrs['polarity'] == "pos" else -1
            precursor = self.lipid_librarian.normalize_precursor_mass(sample.attrs['precursor'])

            additional_features = torch.FloatTensor([[polarity, precursor]])
            features = torch.cat([features, additional_features], dim=0)
            features = features.T

            sample_label, sample_info = self.lipid_librarian.get_regression_label(sample.attrs['lipid_species'])

        return {'features': features, 'label': sample_label, 'info': sample_info, 'dataset_path': torch.IntTensor([index])}

    def get_n_highest_peaks(self, spectrum: np.ndarray, n_peaks: int) -> np.ndarray:
        """Processes the spectrum for a sample in the H5Dataset to prepare it as input for the model.

        Args:
            spectrum (np.ndarray): the spectrum extracted from an HDF5 dataset containing m/z and intensity arrays
            n_peaks (int): maximum number of peaks to be fed into neural network

        Returns:
            np.ndarray: the spectrum containing the number of peaks specified in n_peaks and sorted by descending intensity
        """

        sorted_spectrum = self.bin_spectrum(spectrum)

        # If provided spectrum does not contain sufficient number of peaks, as specified in config, then pad
        if len(sorted_spectrum) < n_peaks:
            diff = n_peaks - len(sorted_spectrum)

            sorted_spectrum = np.pad(array=sorted_spectrum,
                                     pad_width=((0, diff), (0, 0)),
                                     mode="constant",
                                     constant_values=(0, 0))

        return sorted_spectrum[:n_peaks]

    def bin_spectrum(self, spectrum: np.ndarray) -> np.ndarray:
        """Truncates the m/z values at the decimal position defined in the config.yaml and sums up the intensities.

        Args:
            spectrum (np.ndarray): the m/z and intensity values of a dataset from the HDF5 file

        Returns:
            np.ndarray: the binned spectrum ordered by intensity
        """
        mz_array = spectrum[0]
        intensity_array = spectrum[1]

        mz_array_trunc = truncate(mz_array, self.decimal_accuracy)

        mz_intensity_array = pd.DataFrame({'m/z_array': mz_array_trunc, 'intensity_array': intensity_array})
        mz_intensity_array = mz_intensity_array.groupby('m/z_array').intensity_array.sum().reset_index()

        spectrum_trunc = mz_intensity_array.to_numpy()
        sorted_spectrum = spectrum_trunc[np.argsort(-spectrum_trunc[:, 1])]

        return sorted_spectrum


