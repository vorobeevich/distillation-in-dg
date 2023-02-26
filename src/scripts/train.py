import sys
sys.path.append("./")

import argparse

from src.utils.fix_seed import fix_seed
from src.parser.base_parser import BaseParser
from src.trainer.base_trainer import BaseTrainer
# fix random seeds for reproducibility
fix_seed()

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

trainer = BaseTrainer(**BaseParser.parse_config(parser.parse_args()))
trainer.train()