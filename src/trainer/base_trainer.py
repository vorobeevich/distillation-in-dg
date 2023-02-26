import os
from copy import deepcopy
from tqdm import tqdm

import pandas as pd

import torch.utils.data
import torch.nn.functional as F
import torch.nn as nn

import src.datasets.PACS_dataset
from src.utils.init_functions import init_object
from src.parser.base_parser import BaseParser

class BaseTrainer:
    """Class for training the model in the domain generalization mode.
    """

    def __init__(self, config, device, model_config, optimizer_config, scheduler_config, dataset, num_epochs, batch_size, run_id, logger):
        self.config = config

        self.device = device

        self.model_config = model_config

        self.dataset = dataset
        self.domains = dataset["kwargs"]["domain_list"]

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config

        self.num_epochs = num_epochs
        self.batch_size = batch_size

        self.logger = logger

        self.run_id = run_id
        self.checkpoint_dir = "saved/"
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.checkpoint_dir += f"{self.run_id}/"
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)


    def create_loaders(self, test_domain):
        train_dataset, val_dataset, test_dataset = deepcopy(self.dataset), deepcopy(self.dataset), deepcopy(self.dataset) 
        train_dataset["kwargs"]["dataset_type"], val_dataset["kwargs"]["dataset_type"] = ["train"], ["test"]
        test_dataset["kwargs"]["dataset_type"] = ["train", "test"]

        train_dataset["kwargs"]["domain_list"] = [domain for domain in train_dataset["kwargs"]["domain_list"] if domain != self.domains[test_domain]]
        val_dataset["kwargs"]["domain_list"] = [domain for domain in val_dataset["kwargs"]["domain_list"] if domain != self.domains[test_domain]]
        test_dataset["kwargs"]["domain_list"] = [self.domains[test_domain]]
        test_dataset["kwargs"]["augmentations"], val_dataset["kwargs"]["augmentations"] = None, None

        train_dataset = init_object(src.datasets.PACS_dataset, train_dataset)
        val_dataset = init_object(src.datasets.PACS_dataset, val_dataset)
        test_dataset = init_object(src.datasets.PACS_dataset, test_dataset)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=4)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=4)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=4)
        return train_loader, val_loader, test_loader

    def train_epoch_model(self, loader):
        self.model.train()
        accuracy = 0 
        loss_sum = 0
        pbar = tqdm(loader, leave=True)
        for batch in pbar:
            images, labels = batch["image"], batch["label"]
            images, labels = images.to(self.device).float(), labels.to(self.device).long()
            logits = self.model(images)
            loss = self.loss_function(logits, labels)
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
                logits = self.model(images)
                loss = self.loss_function(logits, labels)
                loss_sum += loss.item() * images.shape[0]

                ids = F.softmax(logits, dim=-1).argmax(dim=-1)
                batch_true = (ids == labels).sum() 
                accuracy += batch_true.item()

                pbar.set_description("Accuracy on batch %f loss on batch %f" % ((batch_true / images.shape[0]).item(), loss.item()))

        return accuracy / len(loader.dataset), loss_sum / len(loader.dataset) 

    def train_one_domain(self, test_domain):
        self.model = BaseParser.init_model(self.model_config, self.device)
        train_loader, val_loader, test_loader = self.create_loaders(test_domain)
        self.optimizer = BaseParser.init_optimizer(self.optimizer_config, self.model)
        self.scheduler = BaseParser.init_scheduler(self.scheduler_config, self.optimizer)

        max_train_accuracy, max_val_accuracy, max_test_accuracy = 0, 0, 0
        min_train_loss, min_val_loss, min_test_loss = 1, 1, 1

        for i in range(1, self.num_epochs + 1):
            train_accuracy, train_loss = self.train_epoch_model(train_loader)
            max_train_accuracy, min_train_loss = max(max_train_accuracy, train_accuracy), min(min_train_loss, train_loss)
            self.logger.log_metric(self.domains[test_domain], 'train', 'accuracy', train_accuracy, i)
            self.logger.log_metric(self.domains[test_domain], 'train', 'loss', train_loss, i)

            val_accuracy, val_loss = self.inference_epoch_model(val_loader)
            self.logger.log_metric(self.domains[test_domain], 'val', 'accuracy', val_accuracy, i)
            self.logger.log_metric(self.domains[test_domain], 'val', 'loss', val_loss, i)
            if val_accuracy > max_val_accuracy:
                max_val_accuracy = val_accuracy
                self.save_checkpoint(test_domain)
            max_val_accuracy, min_val_loss = max(max_val_accuracy, val_accuracy), min(min_val_loss, val_loss)

            test_accuracy, test_loss = self.inference_epoch_model(test_loader)
            max_test_accuracy, min_test_loss = max(max_test_accuracy, test_accuracy), min(min_test_loss, test_loss)
            self.logger.log_metric(self.domains[test_domain], 'test', 'accuracy', test_accuracy, i)
            self.logger.log_metric(self.domains[test_domain], 'test', 'loss', test_loss, i)

            self.scheduler.step()

        return max_train_accuracy, max_val_accuracy, max_test_accuracy, \
                min_train_loss, min_val_loss, min_test_loss

    def train(self):       
        train_accuracy, val_accuracy, test_accuracy = [], [], []
        train_loss, val_loss, test_loss = [], [], []
        for i in range(len(self.domains)):
            train_ac, val_ac, test_ac, train_l, val_l, test_l  = \
                    self.train_one_domain(i)
            for metric_list, metric_value in zip([train_accuracy, val_accuracy, test_accuracy, 
                        train_loss, val_loss, test_loss], [train_ac, val_ac, test_ac, train_l, val_l, test_l]):
                metric_list.append(metric_value)
        metrics = {
            'train_accuracy' : train_accuracy,
            'val_accuracy' : val_accuracy,
            'test_accuracy' : test_accuracy,
            'train_loss' : train_loss,
            'val_loss' : val_loss,
            'test_loss' : test_loss,
        }
        metrics = pd.DataFrame(metrics, index=self.domains)
        self.logger.log_table(metrics)

    def save_checkpoint(self, test_domain):
        state = {
            "name": self.config["model"]["name"],
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "config": self.config
        }
        path = f"{self.checkpoint_dir}checkpoint_name_{state['name']}_test_domain_{self.domains[test_domain]}_best.pth"
        torch.save(state, path)
