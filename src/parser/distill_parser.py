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
from src.parser.base_parser import BaseParser
class DistillParser(BaseParser):
    """Static class (a set of methods for which inheritance is possible) for parsing command line arguments 
    and model training configuration.
    """

    @staticmethod
    def parse_config(args: argparse.Namespace) -> dict[str, tp.Any]:
        trainer_params = BaseParser.parse_config(args)
        # read config_yaml from path
        with open(args.config, "r") as stream:
            config = yaml.safe_load(stream)
        trainer_params["model_teacher_config"] = config["model_teacher"]
        for param in ["temperature", "run_id_teacher"]:
            trainer_params[param] = config[param]

        return trainer_params