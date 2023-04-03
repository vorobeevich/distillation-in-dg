import os
from copy import deepcopy
from collections import defaultdict
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

    def __init__(self, config, device, model_config, optimizer_config, scheduler_config, 
                dataset, num_epochs, tracking_step, batch_size, run_id, logger):
        self.config = config

        self.device = device

        self.model_config = model_config

        self.dataset = dataset
        self.domains = dataset["kwargs"]["domain_list"]

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config

        self.num_epochs = num_epochs
        self.tracking_step = tracking_step
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
        pbar = tqdm(loader)
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
            pbar = tqdm(loader)
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
        max_val_accuracy = 0

        for i in range(1, self.num_epochs + 1):
            train_accuracy, train_loss = self.train_epoch_model(train_loader)
            self.logger.log_metric(f"{self.domains[test_domain]}.train.accuracy", train_accuracy, i)
            self.logger.log_metric(f"{self.domains[test_domain]}.train.loss", train_loss, i)

            val_accuracy, val_loss = self.inference_epoch_model(val_loader)
            self.logger.log_metric(f"{self.domains[test_domain]}.val.accuracy", val_accuracy, i)
            self.logger.log_metric(f"{self.domains[test_domain]}.val.loss", val_loss, i)
            if val_accuracy > max_val_accuracy:
                max_val_accuracy = val_accuracy
                self.save_checkpoint(test_domain)

            if i % self.tracking_step == 0:
                test_accuracy, test_loss = self.inference_epoch_model(test_loader)
                self.logger.log_metric(f"{self.domains[test_domain]}.test.accuracy", test_accuracy, i)
                self.logger.log_metric(f"{self.domains[test_domain]}.test.loss", test_loss, i)

            self.scheduler.step()

        self.load_checkpoint(test_domain)
        metrics = dict()
        metrics["train_accuracy"], metrics["train_loss"] = self.train_epoch_model(train_loader)
        metrics["val_accuracy"], metrics["val_loss"] = self.inference_epoch_model(val_loader)
        metrics["test_accuracy"], metrics["test_loss"] = self.inference_epoch_model(test_loader)
        return metrics

    def train(self):       
        # log all info
        all_metrics = defaultdict(list)
        for i in range(len(self.domains)):
            metrics = self.train_one_domain(i)
            for metric in metrics:
                all_metrics[metric].append(metrics[metric])
        all_metrics["domain"] = self.domains
        self.logger.log_table("all_metrics", pd.DataFrame(all_metrics))
        
        # log only val and train accuracies
        accuracy_metrics = dict()
        for ind, domain in enumerate(self.domains):
            accuracy_metrics[f"{domain}_val"] = all_metrics["val_accuracy"][ind]
            accuracy_metrics[f"{domain}_test"] = all_metrics["test_accuracy"][ind]
        self.logger.log_table("results", pd.DataFrame(accuracy_metrics, index=[1]))

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

    def load_checkpoint(self, test_domain):
        self.model = BaseParser.init_model(self.model_config, self.device)
        model_path = f"saved/{self.run_id}/checkpoint_name_{self.model_config['name']}"
        model_path += f"_test_domain_{self.domains[test_domain]}_best.pth"
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint["model"])