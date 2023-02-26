import os
from copy import deepcopy
from tqdm import tqdm

import pandas as pd

import torch.utils.data
import torch.nn.functional as F
import torch.nn as nn

from src.utils.init_functions import init_object
from src.parser.base_parser import BaseParser
from src.trainer.base_trainer import BaseTrainer

class DistillTrainer(BaseTrainer):
    """Class for training the model in the domain generalization mode with distillation.
    """

    def __init__(self, model_teacher_config, run_id_teacher, temperature, **kwargs):
        super().__init__(**kwargs)
        self.run_id_teacher = run_id_teacher
        self.model_teacher_config = model_teacher_config
        self.temperature = temperature
        self.loss_function = nn.KLDivLoss(reduction='batchmean')

    def train_epoch_model(self, loader):
        self.model.train()
        accuracy = 0 
        loss_sum = 0
        pbar = tqdm(loader, leave=True)
        for batch in pbar:
            images, labels = batch["image"], batch["label"]
            images, labels = images.to(self.device).float(), labels.to(self.device).long()
            with torch.inference_mode():
                logits_teacher = self.model_teacher(images)

            logits = self.model(images)
            probs_student, probs_teacher = F.log_softmax(logits, dim=-1), F.softmax(logits_teacher / self.temperature, dim=-1)
            loss = self.loss_function(probs_student, probs_teacher)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_sum += loss.item() * images.shape[0]

            ids = F.softmax(logits, dim=-1).argmax(dim=-1)
            batch_true = (ids == labels).sum() 
            accuracy += batch_true.item()

            pbar.set_description("Accuracy on batch %f loss on batch %f" % ((batch_true / images.shape[0]).item(), loss.item()))

        return accuracy / len(loader.dataset), loss_sum  / len(loader.dataset)

    def inference_epoch_model(self, loader):
        with torch.inference_mode():
            self.model.eval()
            accuracy = 0 
            loss_sum = 0
            pbar = tqdm(loader, leave=True)
            for batch in pbar:
                images, labels = batch["image"].to(self.device), batch["label"].to(self.device).long()
                logits_teacher, logits = self.model_teacher(images), self.model(images)
                probs_student, probs_teacher = F.log_softmax(logits, dim=-1), F.softmax(logits_teacher / self.temperature, dim=-1)
                loss = self.loss_function(probs_student, probs_teacher)
                loss_sum += loss.item() * images.shape[0]

                ids = F.softmax(logits, dim=-1).argmax(dim=-1)
                batch_true = (ids == labels).sum() 
                accuracy += batch_true.item()

                pbar.set_description("Accuracy on batch %f loss on batch %f" % ((batch_true / images.shape[0]).item(), loss.item()))

        return accuracy / len(loader.dataset), loss_sum / len(loader.dataset) 
    
    def train_one_domain(self, test_domain):
        self.load_teacher(test_domain)
        return super().train_one_domain(test_domain)

    def load_teacher(self, test_domain):
        self.model_teacher = BaseParser.init_model(self.model_teacher_config, self.device)
        model_teacher_path = f"saved/{self.run_id_teacher}/checkpoint_name_{self.model_teacher_config['name']}"
        model_teacher_path += f"_test_domain_{self.domains[test_domain]}_best.pth"
        checkpoint = torch.load(model_teacher_path)
        self.model_teacher.load_state_dict(checkpoint["model"])
        for param in self.model_teacher.parameters():
            param.requires_grad = False
