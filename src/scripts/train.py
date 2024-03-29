import sys
sys.path.append("./")

import argparse
import os

parser = argparse.ArgumentParser(description="Train model from config")

parser.add_argument(
    "--config",
    type=str,
    help="Path to config file",
    required=True
)
parser.add_argument(
    "--test",
    nargs="+",
    help="Domains for testing",
    required=True
)
parser.add_argument(
    "--device",
    default="0",
    type=str,
    help="Device index for CUDA_VISIBLE_DEVICES variable"
)
parser.add_argument(
    "--dist",
    help="Will the model train in distillation mode or not",
    action="store_true"
)

args = parser.parse_args()

# set device before src imports (see https://github.com/pytorch/pytorch/issues/9158)
os.environ["CUDA_VISIBLE_DEVICES"] = args.device

from src.trainer.distill_trainer import DistillTrainer
from src.trainer.trainer import Trainer

from src.parser.distill_parser import DistillParser
from src.parser.parser import Parser

# parse args
if args.dist:
    trainer = DistillTrainer(**DistillParser.parse_config(args))
else:
    trainer = Trainer(**Parser.parse_config(args))
trainer.train()
