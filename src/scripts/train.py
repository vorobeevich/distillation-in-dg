import sys
sys.path.append("./")

import argparse
import os

parser = argparse.ArgumentParser(description="Train model from config")

parser.add_argument(
    "--config",
    default=None,
    type=str,
    help="Path to config file",
    required=True
)
parser.add_argument(
    "--device",
    default=None,
    type=str,
    help="Device index for CUDA_VISIBLE_DEVICES variable",
    required=True
)
parser.add_argument(
    "--dist",
    help="Will the model train in distillation mode or not",
    action='store_true'
)

args = parser.parse_args()

# set device before imports (https://github.com/pytorch/pytorch/issues/9158)
os.environ["CUDA_VISIBLE_DEVICES"] = args.device

from src.trainer.distill_trainer import DistillTrainer
from src.trainer.base_trainer import BaseTrainer
from src.parser.distill_parser import DistillParser
from src.parser.base_parser import BaseParser

# parse args
if args.dist:
    trainer = DistillTrainer(**DistillParser.parse_config(args))
else:
    trainer = BaseTrainer(**BaseParser.parse_config(args))
trainer.train()
