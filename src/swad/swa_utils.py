# Burrowed from https://github.com/pytorch/pytorch/blob/master/torch/optim/swa_utils.py
import torch.nn as nn
from copy import deepcopy

class AveragedModel:
    def __init__(self, model):
        self.model = deepcopy(model).cpu()
        self.n = 1

    def forward(self, x):
        return self.model(x)

    def avg_fn(self, averaged_model_parameter, model_parameter):
            return averaged_model_parameter + (model_parameter - averaged_model_parameter) / self.n 

    def update_parameters(self, model):
        """Update averaged model parameters
        Args:
            model: current model to update params
        """
        self.n += 1
        for p_swa, p_model in zip(self.model.parameters(), model.parameters()):
            device = p_swa.device
            p_model_ = p_model.detach().to(device)
            p_swa.detach().copy_(
                self.avg_fn(p_swa.detach(), p_model_)
            )