from tqdm import tqdm

import torch.nn as nn
import torch.nn.functional as F
import torch.distributions.beta
import torch.utils.data

from torchvision.utils import make_grid

from src.parser import Parser
from src.trainer import Trainer
from src.datasets import create_datasets, ImageNet
from src.utils import num_iters_loader


class DistillTrainer(Trainer):
    """Class for training the model in the domain generalization mode with distillation
    """

    def __init__(self, model_teacher_config, run_id_teacher,
                 temperature, image_net, mixup=None, **kwargs):
        super().__init__(**kwargs)
        self.model_teacher_config = model_teacher_config
        self.run_id_teacher = run_id_teacher
        self.image_net = image_net
        self.temperature = temperature
        self.mixup = mixup
        if self.mixup is not None:
            self.lamb = torch.distributions.beta.Beta(self.mixup, self.mixup)
        self.loss_function = nn.KLDivLoss(reduction="batchmean")

    def mixup_on_batch(self, images: torch.Tensor):
        indeces = torch.randperm(images.shape[0])
        shuffled_images = images[indeces]
        lamb = self.lamb.sample()
        images = lamb * images + (1 - lamb) * shuffled_images
        return images

    def process_batch(self, batch):
        images, labels = batch["image"], batch["label"]
        images, labels = images.to(
            self.device).float(), labels.to(
            self.device).long()
        if self.is_logging:
            # logging first batch to wandb every domain
            grid = make_grid(images, nrow=8)
            self.logger.log_image(grid, "First batch at domain")
            self.is_logging = False
        if self.mixup is not None:
            images = self.mixup_on_batch(images)

        with torch.inference_mode():
            logits_teacher = self.model_teacher(images)

        logits = self.model(images)
        probs_student, probs_teacher = F.log_softmax(
            logits / self.temperature, dim=-1), F.softmax(logits_teacher / self.temperature, dim=-1)
        loss = self.loss_function(
            probs_student,
            probs_teacher) * self.temperature * self.temperature

        ids = logits.argmax(dim=-1)
        # if we use mixup then accuracy is not defined, set it to 0
        if self.mixup is not None:
            batch_true = torch.zeros(1)
        else:
            batch_true = (ids == labels).sum()
        return batch_true, loss

    def inference_epoch_model(self, loader):
        mixup = self.mixup
        self.mixup = None
        result = super().inference_epoch_model(loader)
        self.mixup = mixup
        return result

    def create_loaders(self, test_domain, swad: bool = False):
        train_dataset, val_dataset, test_dataset = create_datasets(
            self.dataset, self.domains[test_domain])
        if self.image_net:
            image_net_dataset = ImageNet(len(train_dataset), train_dataset.transforms, train_dataset.augmentations)
            train_dataset = torch.utils.data.ConcatDataset([train_dataset, image_net_dataset])


        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=4)

        if swad:
            train_loader = num_iters_loader(train_loader, self.swad_config["num_iterations"])

        val_loader, test_loader = [torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=4) for dataset in [val_dataset, test_dataset]]
        return train_loader, val_loader, test_loader


    def train_one_domain(self, test_domain):
        self.load_teacher(test_domain)
        super().train_one_domain(test_domain)

    def swad_train_one_domain(self, test_domain):
        self.load_teacher(test_domain)
        super().swad_train_one_domain(test_domain)

    def load_teacher(self, test_domain):
        self.model_teacher = Parser.init_model(
            self.model_teacher_config, self.device)
        model_teacher_path = f'saved/{self.run_id_teacher}/checkpoint_name_{self.model_teacher_config["name"]}'
        model_teacher_path += f"_test_domain_{self.domains[test_domain]}_best.pth"
        checkpoint = torch.load(model_teacher_path)
        self.model_teacher.load_state_dict(checkpoint["model"])
        for param in self.model_teacher.parameters():
            param.requires_grad = False
        self.model_teacher.eval()
