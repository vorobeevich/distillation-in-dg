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
from transformers import AutoImageProcessor, DeiTForImageClassification

from src.datasets import split_dataset
from src.utils import fix_seed, get_device, init_object
from src.logging import WandbLogger


class Parser:
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

        # read config_yaml from path
        with open(args.config, "r") as stream:
            config = yaml.safe_load(stream)

        # fix random seed for reproducibility
        fix_seed(config["seed"])

        # split data on train and test
        split_dataset(config["dataset"]["name"], config["seed"])

        # init params for trainer
        trainer_params = dict()

        trainer_params["test_domains"] = args.test 

        # send some params directly to trainer
        for param in ["dataset", "num_epochs",
                      "batch_size", "run_id", "tracking_step"]:
            trainer_params[param] = config[param]
        for param in ["model", "optimizer", "scheduler", "swad"]:
            trainer_params[f"{param}_config"] = config[param]

        # set seed in dataset config
        trainer_params["dataset"]["kwargs"]["seed"] = config["seed"]

        # init run_id
        if config["run_id"] is None:
            # use timestamp as default run-id
            config["run_id"] = datetime.now().strftime(r"%m%d_%H%M%S")

        # make checkpoint_dir
        if not os.path.exists("saved/"):
            os.makedirs("saved/")
        if not os.path.exists(f'saved/{config["run_id"]}/'):
            os.makedirs(f'saved/{config["run_id"]}/')

        # save device id in config
        config["device_id"] = args.device

        # save yaml version of config
        trainer_params["config"] = config

        # init wandb for logging
        trainer_params["logger"] = WandbLogger(config)

        # init device for training on it
        trainer_params["device"] = get_device()

        # init augmentations (for train images) and transforms (for train and
        # test images)
        trainer_params["dataset"]["kwargs"]["augmentations"] = torchvision.transforms.Compose(
            [init_object(torchvision.transforms, obj_config)
             for obj_config in config["augmentations"]]
        )
        trainer_params["dataset"]["kwargs"]["transforms"] = torchvision.transforms.Compose(
            [init_object(torchvision.transforms, obj_config)
             for obj_config in config["transforms"]]
        )

        return trainer_params

    @staticmethod
    def init_model(config, device):
        # init model params
        if config["name"].startswith("resnet"):
            model = init_object(torchvision.models, config)
            model.fc = nn.Linear(*config["last_layer"])
            model.to(device)
            return model
        else:
            processor = AutoImageProcessor.from_pretrained(config["name"], do_resize=False, do_rescale=False, do_center_crop=False,
                                                           do_normalize=False)
            model = DeiTForImageClassification.from_pretrained(config["name"], num_labels=config["num_labels"]).to(device)
            return model, processor

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
        if config is None:
            return None
        # make deepcopy to not corrupt dict
        scheduler = deepcopy(config)

        # init scheduler with optimizer params
        scheduler["kwargs"].update(optimizer=optimizer)
        scheduler = init_object(torch.optim.lr_scheduler, scheduler)

        return scheduler
