import os
import yaml
import argparse
from datetime import datetime
from copy import deepcopy
import typing as tp

import torchvision.models
import torchvision.transforms
import torch.nn as nn
import torch

from src.logging.wandb import WandbLogger
from src.utils.get_device import get_device
from src.utils.init_functions import init_object
class BaseParser:
    """Static class (a set of methods for which inheritance is possible) for parsing command line arguments 
    and model training configuration.
    """

    @staticmethod
    def parse_config(args: argparse.Namespace) -> dict[str, tp.Any]:
        """Parse command line parameters and config_yaml parameters (dataset, model architecture, 
        training parameters, augmentations, etc).

        Args:
            args (argparse.Namespace): args from cl parser (path to config_yaml, path to checkpoint, GPU device).
        """

        # set device
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    
        # read config_yaml from path
        with open(args.config, "r") as stream:
            config = yaml.safe_load(stream)

        # init params for trainer
        trainer_params = dict()

        # send some params directly to trainer
        for param in ["dataset", "num_epochs", "batch_size", "run_id"]:
            trainer_params[param] = config[param]
        for param in ["model", "optimizer", "scheduler"]:
            trainer_params[f"{param}_config"] = config[param]

        # init run_id
        if config["run_id"] is None:
            # use timestamp as default run-id
            config["run_id"] = datetime.now().strftime(r"%m%d_%H%M%S")

        # save yaml version of config
        trainer_params["config"] = config

        # init wandb for logging    
        trainer_params["logger"] = WandbLogger(config)

        # init device for training on it
        trainer_params["device"] = get_device()
    
        # init augmentations (for train images) and transforms (for train and test images)
        trainer_params["dataset"]["kwargs"]["augmentations"] = torchvision.transforms.Compose(
            [init_object(torchvision.transforms, obj_config) for obj_config in config["augmentations"]]
        )
        trainer_params["dataset"]["kwargs"]["transforms"] = torchvision.transforms.Compose(
            [init_object(torchvision.transforms, obj_config) for obj_config in config["transforms"]]
        )

        return trainer_params

    @staticmethod
    def init_model(config, device):
        # init model params
        model = init_object(torchvision.models, config)
        model.fc = nn.Linear(*config["last_layer"])
        
        # prepare for GPU training
        model.to(device)

        return model
    
    @staticmethod
    def init_optimizer(config, model):
        # make deepcopy to not corrupt dict
        optimizer = deepcopy(config)

        # init optimizer with model params
        optimizer["kwargs"].update(params=model.parameters())
        optimizer = init_object(torch.optim, optimizer)

        return optimizer

    @staticmethod
    def init_scheduler(config, optimizer):
        # make deepcopy to not corrupt dict
        scheduler = deepcopy(config)

        # init scheduler with optimizer params
        scheduler["kwargs"].update(optimizer=optimizer)
        scheduler = init_object(torch.optim.lr_scheduler, scheduler)

        return scheduler
