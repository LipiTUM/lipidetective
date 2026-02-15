from __future__ import annotations

import argparse
import logging
import os
from typing import Any

import h5py
import numpy as np
from pyteomics import mzml


def parse_arguments() -> tuple[str, str]:
    parser = argparse.ArgumentParser(
        description="This script takes in a folder containing mzML files and prepares them "
        "for processing with LipiDetective as HDF5 files."
    )

    parser.add_argument(
        "-i", "--mzml_dir", help="path to directory containing mzML files", required=True
    )
    parser.add_argument(
        "-o", "--output_dir", help="path to output directory for HDF5 file ", required=True
    )
    arguments = parser.parse_args()

    return arguments.mzml_dir, arguments.output_dir


def add_spectrum_to_hdf5(
    spectrum: dict[str, Any], file_number: int, file_name: str, hdf5_file: h5py.File
) -> None:
    mz = spectrum["m/z array"]
    intensity = spectrum["intensity array"]
    intensity = intensity / intensity.max()

    peaks = np.stack((mz, intensity), axis=0)
    bool_peaks = peaks[1, :] != 0
    peaks = peaks[:, bool_peaks]
    peaks = peaks[:, (50 <= peaks[0]) & (peaks[0] < 1600)]

    if peaks.size:
        mode = "pos" if "positive scan" in spectrum else "neg"

        lipid_name = "nist_spectrum"

        dataset_name = f"{lipid_name} | {mode} | nist | {file_number}{spectrum['index']}"

        dataset = hdf5_file.create_dataset(f"/all_datasets/{dataset_name}", data=peaks)

        dataset.attrs["lipid_species"] = ""
        dataset.attrs["adduct"] = ""
        dataset.attrs["scan_index"] = spectrum["index"]
        dataset.attrs["level"] = spectrum["ms level"]
        dataset.attrs["precursor"] = float(
            spectrum["precursorList"]["precursor"][0]["selectedIonList"]["selectedIon"][0][
                "selected ion m/z"
            ]
        )
        dataset.attrs["polarity"] = mode
        dataset.attrs["source"] = file_name


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%d/%m/%Y - %H:%M:%S",
    )

    input_dir, output_dir = parse_arguments()

    all_filepaths = []

    for file in os.listdir(input_dir):
        if file != ".DS_Store":
            file_path = os.path.join(input_dir, file)
            all_filepaths.append(file_path)

    hdf5_file_name = f"{input_dir.split('/')[-1]}.hdf5"
    hdf5_file_path = os.path.join(output_dir, hdf5_file_name)
    hdf5_file = h5py.File(hdf5_file_path, "w")
    hdf5_file.create_group("/all_datasets")

    logging.info("Processing:")
    for idx, file in enumerate(all_filepaths):
        file_name = file.split("/")[-1]
        logging.info(file_name)

        spectra = list(mzml.read(file))
        for spectrum in spectra:
            if spectrum["ms level"] == 2:
                add_spectrum_to_hdf5(spectrum, idx, file_name, hdf5_file)

    logging.info(f"HDF5 file contains {len(hdf5_file['/all_datasets'])} processed MS2 spectra.")

    hdf5_file.close()

    logging.info(f"HDF5 file saved at {hdf5_file_path}")
