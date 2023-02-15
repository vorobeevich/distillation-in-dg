import os
import yaml
import argparse
from datetime import datetime
from copy import deepcopy

import torch.optim
import torch.nn as nn
import torchvision.models
import torchvision.transforms

from src.utils.get_device import get_device
from src.utils.init_functions import init_object, init_object_list
from src.trainer.trainer import Trainer
from src.logging.wandb import WandbLogger

def parse_config(args: argparse.Namespace):
    """Parse command line parameters and config_yaml parameters (dataset, model architecture, 
    training parameters, augmentations).

    Args:
        args (argparse.Namespace): args from cl parser (path to config_yaml, path to checkpoint, GPU device).
    """

    # set device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    
    # read config_yaml from path
    with open(args.config, "r") as stream:
        config_yaml = yaml.safe_load(stream)

    # init trainer params
    config = dict()
    for param in config_yaml["trainer"]:
        config[param] = config_yaml["trainer"][param]

    # save yaml version of config
    config["config"] = deepcopy(config_yaml)

    # init run_id
    if config["run_id"] is None:
        # use timestamp as default run-id
        config["run_id"] = datetime.now().strftime(r"%m%d_%H%M%S")

    # init wandb for logging    
    config["logger"] = WandbLogger(config_yaml)

    # init device
    config["device"] = get_device()

    # copy params
    config["model"] = config_yaml["model"]
    config["optimizer"] = config_yaml["optimizer"]
    config["scheduler"] = config_yaml["scheduler"]
    
    # init augmentations (to train images) and transforms (to train and test images)
    config["dataset"]["kwargs"]["augmentations"] = torchvision.transforms.Compose(init_object_list(torchvision.transforms, config_yaml["augmentations"]))
    config["dataset"]["kwargs"]["transforms"] = torchvision.transforms.Compose(init_object_list(torchvision.transforms, config_yaml["transforms"]))


    trainer = Trainer(**config)
    return trainer