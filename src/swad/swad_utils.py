import torch.nn as nn
import numpy as np
import queue
import torch.nn as nn

from src.swad.swa_utils import AveragedModel

class SWAD:
    def __init__(self, n_converge: int, n_tolerance: int, tolerance_ratio: float):
        self.n_converge = n_converge
        self.n_tolerance = n_tolerance
        self.tolerance_ratio = tolerance_ratio
        self.val_loss = []
        self.models = queue.Queue()
        self.average_begin = False
        self.average_finish = False

    def update(self, loss: float, model: nn.Module):
        if self.average_finish:
            return
        self.val_loss.append(loss)
        self.models.put(model) 

        if not self.average_begin:
            if self.models.qsize() <= self.n_converge:
                return
            if min(self.val_loss[-self.n_converge:]) == self.val_loss[-self.n_converge]:
                self.final_model = AveragedModel(self.models.get())
                self.average_begin = True
                self.loss_threshold = np.mean(self.val_loss[-self.n_converge:]) * self.tolerance_ratio
                print("BEGIN, IND: ", len(self.val_loss) - self.n_converge + 1, "THRESHOLD: ", self.loss_threshold)
            else:
                self.models.get()
        else:
            if self.models.qsize() < self.n_tolerance - 1:
                return
            self.final_model.update_parameters(self.models.get())
            if min(self.val_loss[-self.n_tolerance:]) > self.loss_threshold:
                self.average_finish = True
                while self.models.qsize() > 0:
                    self.models.get()
                print("END, IND: ", len(self.val_loss) - self.n_tolerance + 1)
        
    def finish(self):
        while not self.average_finish and self.models.qsize() > 0:
            self.final_model.update_parameters(self.models.get())