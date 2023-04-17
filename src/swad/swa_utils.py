# Burrowed from https://github.com/pytorch/pytorch/blob/master/torch/optim/swa_utils.py
# modified for the DomainBed.
import torch.nn as nn
from copy import deepcopy

class AveragedModel(nn.Module):
    def __init__(self, model):
        super(AveragedModel, self).__init__()
        self.model = deepcopy(model)

        def avg_fn(averaged_model_parameter, model_parameter):
            return averaged_model_parameter + (model_parameter - averaged_model_parameter) / self.n 
    
        self.avg_fn = avg_fn
        self.n = 1

    def forward(self, x):
        return self.model(x)

    def update_parameters(self, model):
        """Update averaged model parameters
        Args:
            model: current model to update params
        """
        self.n += 1
        for p_swa, p_model in zip(self.parameters(), model.parameters()):
            device = p_swa.device
            p_model_ = p_model.detach().to(device)
            p_swa.detach().copy_(
                self.avg_fn(p_swa.detach(), p_model_,)
            )