import json
import pandas as pd
import numpy as np
import torch

from pyteomics import mzml
from torch.utils.data import Dataset

from src.lipidetective.helpers.utils import truncate


class PredictionDataset(Dataset):

    def __init__(self, file_path, config):
        self.file_name = file_path.split('/')[-1]
        self.file_path = file_path
        self.config = config
        self.file = self.process_input()
        self.dataset_len = len(self.file)

    def __len__(self) -> int:
        return self.dataset_len

    def __getitem__(self, index: int) -> dict:
        sample = self.file[index]
        features = self.get_n_highest_peaks(sample['mz'], sample['intensity'], sample['precursor'])
        spectrum_info = {'index': sample['index'], 'file': self.file_name, 'polarity': sample['polarity'], 'precursor': "{0:.2f}".format(sample['precursor'])}

        return {'features': features, 'info': spectrum_info}

    def process_input(self):
        if self.file_path.endswith('.mzML'):
            return self.process_mzml()
        elif self.file_path.endswith('.json'):
            return self.process_json()
        else:
            return None

    def process_mzml(self):
        spectra = list(mzml.read(self.file_path))
        ms2_spectra = []
        for spectrum in spectra:
            if spectrum['ms level'] == 2:
                spectrum_id = spectrum['index']
                precursor = float(spectrum['precursorList']['precursor'][0]['selectedIonList']['selectedIon'][0]['selected ion m/z'])
                mz_array = spectrum['m/z array']
                intensity_array = spectrum['intensity array']
                polarity = '+' if 'positive scan' in spectrum else ('-' if 'negative scan' in spectrum else None)
                spectrum_entry = {'index': spectrum_id, 'precursor': precursor, 'mz': mz_array, 'intensity': intensity_array, 'polarity': polarity}
                ms2_spectra.append(spectrum_entry)
        return ms2_spectra

    def process_json(self):
        with open(self.file_path, 'r') as file:
            spectra = json.load(file)
        return spectra

    def get_n_highest_peaks(self, mz_array, intensity_array, precursor):
        n_peaks = self.config['input_embedding']['n_peaks']
        decimal_accuracy = self.config['input_embedding']['decimal_accuracy']

        mz_array_trunc = truncate(mz_array, decimal_accuracy)
        mz_intensity_array = pd.DataFrame({'m/z_array': mz_array_trunc, 'intensity_array': intensity_array})
        mz_intensity_array = mz_intensity_array.groupby('m/z_array').intensity_array.sum().reset_index()

        spectrum_trunc = mz_intensity_array.to_numpy()
        sorted_spectrum = spectrum_trunc[np.argsort(-spectrum_trunc[:, 1])]

        if len(sorted_spectrum) < n_peaks:
            diff = n_peaks - len(sorted_spectrum)

            sorted_spectrum = np.pad(array=sorted_spectrum,
                                     pad_width=((0, diff), (0, 0)),
                                     mode="constant",
                                     constant_values=(0, 0))

        features = sorted_spectrum[:n_peaks]

        features = features[:, 0]
        features = torch.IntTensor(features * (10 ** decimal_accuracy))

        precursor_mz = int(float(precursor) * (10 ** decimal_accuracy))

        if precursor_mz not in features:
            features[-1] = precursor_mz

        return features

