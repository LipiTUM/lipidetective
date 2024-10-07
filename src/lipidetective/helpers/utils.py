import argparse
import yaml
import traceback
import torch
import os
import logging
import math
import numpy as np
import random


def parse_config():
    parser = argparse.ArgumentParser(description='This script generates a deep learning model for lipid mass spectra.')
    parser.add_argument('--config', help='path to config file containing all parameters', required=True)
    parser.add_argument('--head_node_ip', help='ip address of head node if running tune on cluster', required=False)
    arguments = parser.parse_args()

    return read_yaml(arguments.config), arguments


def read_yaml(file_to_open: str):
    try:
        with open(file_to_open, 'r') as file:
            loaded_file = yaml.safe_load(file)
        return loaded_file
    except Exception as e:
        traceback.print_exc()


def write_yaml(file_to_open, dict_to_write):
    try:
        with open(file_to_open, 'w') as file:
            yaml.dump(dict_to_write, file)
    except Exception as e:
        traceback.print_exc()


def set_device(config):
    if torch.cuda.is_available():
        if config['cuda']['gpu_nr'] is None:
            device = torch.device("cuda:{GPU}".format(GPU="0"))
        else:
            device = torch.device("cuda:{GPU}".format(GPU=config['gpu_nr']))

        if config['tune']['fractional_gpu']:
            nr_gpus = 0.5
        else:
            nr_gpus = 1
        logging.info("Running LipiDetective on the GPU")

    else:
        device = torch.device("cpu")
        nr_gpus = 0
        logging.info("Running LipiDetective on the CPU")

    return device, nr_gpus


def set_seeds(seed: int = 42) -> None:
    torch.set_float32_matmul_precision('high')

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.enabled = False

    logging.info(f"Random seed set as {seed}")


def is_main_process():
    """This function is only necessary when running LipiDetective using tune and makes sure that certain processes
    are only performed once per run.
    """
    return 'LOCAL_RANK' not in os.environ.keys() and 'NODE_RANK' not in os.environ.keys()


def truncate(values: np.ndarray, decimal_places=0):
    factor = 10 ** decimal_places
    return [(math.floor(x * factor) / factor) for x in values]


def is_lipid_class_with_slash(lipid_name):
    return lipid_name.startswith(('Cer', 'SM', 'EPC', 'IPC', 'HexCer', 'LacCer', 'GalCer', 'SHexCer', 'GlcCer', 'SE', 'Hex2Cer',
                         'Hex3Cer', 'FAHFA', 'GM1', 'GD1a', 'GD1b', 'GT1b', 'GM3'))

