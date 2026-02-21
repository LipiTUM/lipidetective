"""Pre-compute transformer/LSTM features from a raw HDF5 dataset.

Reads every spectrum, runs binning + top-N peak selection + integer m/z
conversion, and writes a Parquet file with pre-computed features and the
original metadata preserved.  The resulting file is small, fast to load,
safe to share (no pickle), and compatible with HuggingFace Datasets.

Usage:
    uv run python -m lipidetective.workflow.precompute_dataset \
        --input data/processed/merged_dataset.hdf5 \
        --output data/processed/merged_dataset_precomputed.parquet \
        --n_peaks 30 --max_mz 1600 --decimal_accuracy 1
"""

from __future__ import annotations

import argparse
import logging

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lipidetective.helpers.utils import truncate


def precompute_dataset(
    input_path: str,
    output_path: str,
    n_peaks: int,
    max_mz: int,
    decimal_accuracy: int,
) -> None:
    """Read raw spectra and write pre-computed features to Parquet."""
    max_index = max_mz * 10**decimal_accuracy

    rows: list[dict[str, object]] = []

    with h5py.File(input_path, "r") as src:
        src_group = src["all_datasets"]
        names = list(src_group.keys())

        logging.info(f"Pre-computing {len(names)} spectra from {input_path}")

        for i, name in enumerate(names):
            sample = src_group[name]
            spectrum: np.ndarray = sample[()]

            # Bin spectrum: truncate m/z, group duplicates, sum intensities
            mz_array_trunc = truncate(spectrum[0], decimal_accuracy)
            unique_mz, inverse = np.unique(mz_array_trunc, return_inverse=True)
            summed_intensities = np.bincount(inverse, weights=spectrum[1])
            binned = np.column_stack((unique_mz, summed_intensities))
            sorted_spectrum = binned[np.argsort(-binned[:, 1])]

            # Take top N peaks, pad if needed
            if len(sorted_spectrum) < n_peaks:
                diff = n_peaks - len(sorted_spectrum)
                sorted_spectrum = np.pad(sorted_spectrum, ((0, diff), (0, 0)), mode="constant")
            peaks = sorted_spectrum[:n_peaks]

            # Convert to integer m/z indices
            mz_values = peaks[:, 0]
            features = np.rint(mz_values * (10**decimal_accuracy)).astype(np.int64)

            # Filter out peaks outside embedding vocab range
            features[features >= max_index] = 0

            # Insert precursor m/z if not already present
            precursor_mz = int(round(float(sample.attrs["precursor"]) * (10**decimal_accuracy)))
            if precursor_mz < max_index and precursor_mz not in features:
                features[-1] = precursor_mz

            row: dict[str, object] = {
                "dataset_name": name,
                "features": features.tolist(),
                "lipid_species": str(sample.attrs["lipid_species"]),
                "adduct": str(sample.attrs["adduct"]),
                "polarity": str(sample.attrs["polarity"]),
                "precursor": float(sample.attrs["precursor"]),
            }
            rows.append(row)

            if (i + 1) % 10000 == 0:
                logging.info(f"  Processed {i + 1}/{len(names)} spectra")

    # Write to Parquet
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, output_path, compression="zstd")
    logging.info(f"Done. Written {len(rows)} spectra to {output_path}")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Pre-compute features for transformer/LSTM")
    parser.add_argument("--input", required=True, help="Path to raw HDF5 dataset")
    parser.add_argument("--output", required=True, help="Path for output Parquet file")
    parser.add_argument("--n_peaks", type=int, default=30)
    parser.add_argument("--max_mz", type=int, default=1600)
    parser.add_argument("--decimal_accuracy", type=int, default=1)
    args = parser.parse_args()

    precompute_dataset(args.input, args.output, args.n_peaks, args.max_mz, args.decimal_accuracy)


if __name__ == "__main__":
    main()
