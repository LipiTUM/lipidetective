from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from pyteomics import mzml
from torch.utils.data import Dataset

from lipidetective.helpers.utils import truncate


class PredictionDataset(Dataset[dict[str, Any]]):
    def __init__(self, file_path: str, config: dict[str, Any]) -> None:
        self.file_name: str = Path(file_path).name
        self.file_path: str = file_path
        self.config: dict[str, Any] = config
        self.file: list[dict[str, Any]] = self.process_input()
        self.dataset_len: int = len(self.file)

    def __len__(self) -> int:
        return int(self.dataset_len)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.file[index]
        features = self.get_n_highest_peaks(sample["mz"], sample["intensity"], sample["precursor"])
        spectrum_info = {
            "index": sample["index"],
            "file": self.file_name,
            "polarity": sample["polarity"],
            "precursor": f"{sample['precursor']:.2f}",
        }

        return {"features": features, "info": spectrum_info}

    def process_input(self) -> list[dict[str, Any]]:
        if self.file_path.endswith(".mzML"):
            return self.process_mzml()
        elif self.file_path.endswith(".json"):
            return self.process_json()
        else:
            raise ValueError(f"Unsupported format. Expected .mzML or .json, got: {self.file_path}")

    def process_mzml(self) -> list[dict[str, Any]]:
        spectra = list(mzml.read(self.file_path))
        ms2_spectra: list[dict[str, Any]] = []
        for spectrum in spectra:
            if spectrum["ms level"] == 2:
                spectrum_id: int = spectrum["index"]
                precursor: float = float(
                    spectrum["precursorList"]["precursor"][0]["selectedIonList"]["selectedIon"][0][
                        "selected ion m/z"
                    ]
                )
                mz_array: np.ndarray = spectrum["m/z array"]
                intensity_array: np.ndarray = spectrum["intensity array"]
                polarity: str | None = (
                    "+"
                    if "positive scan" in spectrum
                    else ("-" if "negative scan" in spectrum else None)
                )
                spectrum_entry: dict[str, Any] = {
                    "index": spectrum_id,
                    "precursor": precursor,
                    "mz": mz_array,
                    "intensity": intensity_array,
                    "polarity": polarity,
                }
                ms2_spectra.append(spectrum_entry)
        return ms2_spectra

    def process_json(self) -> list[dict[str, Any]]:
        with open(self.file_path) as file:
            spectra: list[dict[str, Any]] = json.load(file)
        return spectra

    def get_n_highest_peaks(
        self, mz_array: np.ndarray, intensity_array: np.ndarray, precursor: float
    ) -> torch.Tensor:
        n_peaks = self.config["input_embedding"]["n_peaks"]
        decimal_accuracy = self.config["input_embedding"]["decimal_accuracy"]

        mz_array_trunc = truncate(mz_array, decimal_accuracy)
        mz_intensity_array = pd.DataFrame(
            {"m/z_array": mz_array_trunc, "intensity_array": intensity_array}
        )
        mz_intensity_array = (
            mz_intensity_array.groupby("m/z_array").intensity_array.sum().reset_index()
        )

        spectrum_trunc = mz_intensity_array.to_numpy()
        sorted_spectrum = spectrum_trunc[np.argsort(-spectrum_trunc[:, 1])]

        if len(sorted_spectrum) < n_peaks:
            diff = n_peaks - len(sorted_spectrum)

            sorted_spectrum = np.pad(
                array=sorted_spectrum,
                pad_width=((0, diff), (0, 0)),
                mode="constant",
                constant_values=(0, 0),
            )

        spectrum_peaks = sorted_spectrum[:n_peaks]
        mz_values = spectrum_peaks[:, 0]
        features = torch.IntTensor(np.rint(mz_values * (10**decimal_accuracy)))

        # Filter out peaks with m/z >= max_mz (outside embedding vocab range)
        max_index = self.config["input_embedding"]["max_mz"] * 10**decimal_accuracy
        features = features[features < max_index]
        if len(features) < n_peaks:
            features = torch.nn.functional.pad(features, (0, n_peaks - len(features)))

        precursor_mz = int(round(float(precursor) * (10**decimal_accuracy)))

        if precursor_mz < max_index and precursor_mz not in features:
            features[-1] = precursor_mz

        return features
