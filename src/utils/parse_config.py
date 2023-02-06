import os
import yaml
import argparse

import torch
import torchvision

from src.utils.get_device import get_device
import src.datasets.PACS_dataset

def init_object(obj_cfg, module):
    return getattr(module, obj_cfg["name"])(**obj_cfg["kwargs"])

def init_object_list(cfg_list, module):
    res = []
    for obj in cfg_list:
        res.append(init_object(obj, module))
    return res

def parse_config(args: argparse.Namespace):
    """Parse command line parameters and config parameters (dataset, model architecture, 
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
        "num_epochs" : cfg["trainer"]["num_epochs"]
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
    
    # init augmentations (to train images) and transforms (to train and test images)
    trainer_params["augmentations"] = torchvision.transforms.Compose(init_object_list(cfg["augmentations"], torchvision.transforms))
    trainer_params["transforms"] = torchvision.transforms.Compose(init_object_list(cfg["transforms"], torchvision.transforms))

    return trainer_params