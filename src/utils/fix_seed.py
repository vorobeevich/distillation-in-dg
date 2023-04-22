import random
import os
import torch
import numpy as np


def fix_seed(seed: int = 42) -> None:
    """Make the random behavior of some functions deterministic for reproducibility of experiments.
    Warning: this function does not provide complete deterministic training of the neural network.
     If you are using some functions (like sklearn library) you must manually pass the seed to functions.

    Args:
        seed (int, optional): seed for fixation. Defatults to 42.
    """

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
