import os
from typing import ClassVar
import torch
import numpy as np
import random
import os
import yaml
from omegaconf import OmegaConf
import importlib
import argparse
DIR_PATH = os.path.dirname(os.path.realpath(__file__))

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def load_args(filename, args):
    with open(filename, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    for key, group in data_loaded.items():
        for key, val in group.items():
            setattr(args, key, val)

def calculate_params(model):
    all_para = sum([np.prod(list(p.size())) for p in model.parameters()])
    requires_para = sum(
        [np.prod(list(p.size())) for p in model.parameters() if p.requires_grad]
    )
    type_size = 4
    all_para_size = all_para * type_size / 1024 / 1024
    requires_para_size = requires_para * type_size / 1024 / 1024
    return all_para, requires_para, all_para_size, requires_para_size


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0", "none", "null"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def path2optional_bool(v):
    if v.lower() in ("no", "false", "f", "n", "0", "none", "null"):
        return False
    else:
        return v


def find_index(arr, value):
    try:
        return arr.index(value)
    except ValueError:
        return -1
