import os
import yaml
import argparse
import importlib

import torch
import torchvision

from src.utils.get_device import get_device

def init_object(obj_cfg, module):
    return getattr(module, obj_cfg["name"])(**obj_cfg["kwargs"])

def parse_config(args: argparse.Namespace):
    """Function to parse command line parameters and config parameters (dataset, model architecture, 
    training parameters, augmentations).

    Args:
        args (argparse.Namespace): args from cl parser (path to config, path to checkpoint, GPU device).
    """

    # set device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = get_device()

    # read config from path
    with open(args.config, "r") as stream:
        cfg = yaml.safe_load(stream)

    # init return params
    trainer_params = {
        'num_epochs' : cfg["trainer"]["num_epochs"]
    }
    # init model params
    trainer_params["model"] = init_object(cfg["model"], torchvision.models)
    trainer_params["model"].fc = torch.nn.Linear(*cfg["last_layer"])
    
    # prepare for GPU training
    trainer_params["model"].to(device)
    
    # init optimizer with model params
    cfg["optimizer"]["kwargs"].update(params=trainer_params["model"].parameters())
    trainer_params["optimizer"] = init_object(cfg["optimizer"], torch.optim)
    
    # init scheduler with optimizer params
    cfg["scheduler"]["kwargs"].update(optimizer=trainer_params["optimizer"])
    trainer_params["scheduler"] = init_object(cfg["scheduler"], torch.optim.lr_scheduler)
    
    trainer_params["augmentations"] = []
    for aug in cfg["augmentations"]:
        trainer_params["augmentations"].append(init_object(aug, torchvision.transforms))
    trainer_params["augmentations"] = torchvision.transforms.Compose(trainer_params["augmentations"])
    return trainer_params