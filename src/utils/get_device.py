import torch

def get_device(gpu_usage: bool = True) -> torch.device:
    """Return the GPU device on which the code will run (if gpu_usage), otherwise the code will be run on the CPU.

    Args:
        gpu_usage (bool, optional): should the programm use GPU or not. Defatults to True.

    Returns:
        torch.device: the device on which the code will run.
    """

    if gpu_usage and torch.cuda.device_count() == 0:
        print(
            "Warning: There's no GPU available on this machine,"
            "training will be performed on CPU."
        )
        device = "cpu"
    elif not gpu_usage:
        device = "cpu"
    else:
        device = torch.device("cuda:0")

    return device
