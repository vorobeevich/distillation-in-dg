from tqdm import tqdm

import torch.nn as nn
import torch.nn.functional as F
import torch.distributions.beta

from torchvision.utils import make_grid

from src.parser.base_parser import BaseParser
from src.trainer.base_trainer import BaseTrainer

class DistillTrainer(BaseTrainer):
    """Class for training the model in the domain generalization mode with distillation.
    """

    def __init__(self, model_teacher_config, run_id_teacher, temperature, mixup=None, **kwargs):
        super().__init__(**kwargs)
        self.run_id_teacher = run_id_teacher
        self.model_teacher_config = model_teacher_config
        self.temperature = temperature
        self.mixup = mixup
        if self.mixup is not None:
            self.lamb = torch.distributions.beta.Beta(self.mixup, self.mixup)
        self.loss_function = nn.KLDivLoss(reduction='batchmean')
    
    def mixup_on_batch(self, images: torch.Tensor):
        indeces = torch.randperm(images.shape[0])
        shuffled_images = images[indeces]
        lamb = self.lamb.sample()
        images = lamb * images + (1 - lamb) * shuffled_images
        return images

    def train_epoch_model(self, loader):
        self.model.train()
        accuracy = 0 
        loss_sum = 0
        pbar = tqdm(enumerate(loader), leave=True)
        for ind, batch in pbar:
            images, labels = batch["image"], batch["label"]
            images, labels = images.to(self.device).float(), labels.to(self.device).long()

            if self.mixup is not None:
                images = self.mixup_on_batch(images)
                if ind == 0:
                    grid = make_grid(images, nrow=8)
                    self.logger.log_image(grid, "Mixup_batch")                

            with torch.inference_mode():
                logits_teacher = self.model_teacher(images)

            logits = self.model(images)
            probs_student, probs_teacher = F.log_softmax(logits / self.temperature, dim=-1), F.softmax(logits_teacher / self.temperature, dim=-1)
            loss = self.loss_function(probs_student, probs_teacher) * self.temperature * self.temperature

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_sum += loss.item() * images.shape[0]

            ids = F.softmax(logits, dim=-1).argmax(dim=-1)
            if self.mixup is not None:
                batch_true = (ids == ids).sum()
            else: 
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
                probs_student, probs_teacher = F.log_softmax(logits / self.temperature, dim=-1), F.softmax(logits_teacher / self.temperature, dim=-1)
                loss = self.loss_function(probs_student, probs_teacher) * self.temperature * self.temperature
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
        self.model_teacher.eval()
