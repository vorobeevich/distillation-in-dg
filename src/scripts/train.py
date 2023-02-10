import sys
sys.path.append("./")

import argparse

from src.utils.fix_seed import fix_seed
from src.utils.parse_config import parse_config
from src.trainer.trainer import Trainer
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

trainer = parse_config(parser.parse_args())
trainer.train()