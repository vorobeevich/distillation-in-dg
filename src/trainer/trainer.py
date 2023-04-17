import os
from copy import deepcopy
from collections import defaultdict
import queue
from tqdm import tqdm

import pandas as pd
import numpy as np

import torch.utils.data
import torch.nn.functional as F
import torch.nn as nn

import src.datasets.PACS_dataset
from src.utils.init_functions import init_object
from src.parser.parser import Parser
from src.swad.swa_utils import AveragedModel

def check(model, loader, loss_function):
    with torch.inference_mode():
        model.eval()
        accuracy = 0
        loss_sum = 0
        pbar = tqdm(loader)
        for batch in pbar:
            images, labels = batch["image"], batch["label"]
            images, labels = images.to(
                "cuda").float(), labels.to(
                "cuda").long()
            logits = model(images)

            loss = loss_function(logits, labels)

            ids = F.softmax(logits, dim=-1).argmax(dim=-1)
            batch_true = (ids == labels).sum()
            loss_sum += loss.item() * batch["image"].shape[0]
            accuracy += batch_true.item()

            pbar.set_description(
                "Accuracy on batch %f loss on batch %f" %
                ((batch_true / batch["image"].shape[0]).item(), loss.item()))

        return accuracy / len(loader.dataset), loss_sum / len(loader.dataset)

class Trainer:
    """Class for training the model in the domain generalization mode.
    """

    def __init__(self, config, device, model_config, optimizer_config, scheduler_config,
                swad_config, dataset, num_epochs, tracking_step, batch_size, run_id, logger):
        self.config = config

        self.device = device

        self.model_config = model_config

        self.dataset = dataset
        self.domains = dataset["kwargs"]["domain_list"]

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.swad_config = swad_config
        if self.swad_config is not None:
            self.swad_config["average_begin"] = False
            self.swad_config["average_finish"] = False

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
        train_dataset, val_dataset, test_dataset = deepcopy(
            self.dataset), deepcopy(
            self.dataset), deepcopy(
            self.dataset)
        train_dataset["kwargs"]["dataset_type"], val_dataset["kwargs"]["dataset_type"] = [
            "train"], ["test"]
        test_dataset["kwargs"]["dataset_type"] = ["train", "test"]

        train_dataset["kwargs"]["domain_list"] = [
            domain for domain in train_dataset["kwargs"]["domain_list"] if domain != self.domains[test_domain]]
        val_dataset["kwargs"]["domain_list"] = [
            domain for domain in val_dataset["kwargs"]["domain_list"] if domain != self.domains[test_domain]]
        test_dataset["kwargs"]["domain_list"] = [self.domains[test_domain]]
        test_dataset["kwargs"]["augmentations"], val_dataset["kwargs"]["augmentations"] = None, None

        train_dataset = init_object(src.datasets.PACS_dataset, train_dataset)
        val_dataset = init_object(src.datasets.PACS_dataset, val_dataset)
        test_dataset = init_object(src.datasets.PACS_dataset, test_dataset)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=4)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=4)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=4)
        return train_loader, val_loader, test_loader

 
    def process_batch(self, batch):
        images, labels = batch["image"], batch["label"]
        images, labels = images.to(
            self.device).float(), labels.to(
            self.device).long()
        logits = self.model(images)

        loss = self.loss_function(logits, labels)

        ids = F.softmax(logits, dim=-1).argmax(dim=-1)
        batch_true = (ids == labels).sum()

        return batch_true, loss
    
    def init_training(self):
        self.model = Parser.init_model(self.model_config, self.device)
        self.optimizer = Parser.init_optimizer(
            self.optimizer_config, self.model)
        self.scheduler = Parser.init_scheduler(
            self.scheduler_config, self.optimizer)
    
    def swad_train_regime(self):
        self.model.train()
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    def train_epoch_model(self, loader):
        self.model.train()
        accuracy = 0
        loss_sum = 0
        pbar = tqdm(loader)
        for batch in pbar:
            batch_true, loss = self.process_batch(batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_sum += loss.item() * batch["image"].shape[0]
            accuracy += batch_true.item()

            pbar.set_description(
                "Accuracy on batch %f loss on batch %f" %
                ((batch_true / batch["image"].shape[0]).item(), loss.item()))
        return accuracy / len(loader.dataset), loss_sum / len(loader.dataset)

    def inference_epoch_model(self, loader):
        with torch.inference_mode():
            self.model.eval()
            accuracy = 0
            loss_sum = 0
            pbar = tqdm(loader)
            for batch in pbar:
                batch_true, loss = self.process_batch(batch)
                loss_sum += loss.item() * batch["image"].shape[0]
                accuracy += batch_true.item()

                pbar.set_description(
                    "Accuracy on batch %f loss on batch %f" %
                    ((batch_true / batch["image"].shape[0]).item(), loss.item()))

        return accuracy / len(loader.dataset), loss_sum / len(loader.dataset)

    def train_one_domain(self, test_domain):
        train_loader, val_loader, test_loader = self.create_loaders(
            test_domain)
        max_val_accuracy = 0

        for i in range(1, self.num_epochs + 1):
            train_accuracy, train_loss = self.train_epoch_model(train_loader)
            self.logger.log_metric(
                f"{self.domains[test_domain]}.train.accuracy", train_accuracy, i)
            self.logger.log_metric(
                f"{self.domains[test_domain]}.train.loss", train_loss, i)

            val_accuracy, val_loss = self.inference_epoch_model(val_loader)
            self.logger.log_metric(
                f"{self.domains[test_domain]}.val.accuracy", val_accuracy, i)
            self.logger.log_metric(
                f"{self.domains[test_domain]}.val.loss", val_loss, i)
            if val_accuracy > max_val_accuracy:
                max_val_accuracy = val_accuracy
                self.save_checkpoint(test_domain)

            if i % self.tracking_step == 0:
                test_accuracy, test_loss = self.inference_epoch_model(
                    test_loader)
                self.logger.log_metric(
                    f"{self.domains[test_domain]}.test.accuracy", test_accuracy, i)
                self.logger.log_metric(
                    f"{self.domains[test_domain]}.test.loss", test_loss, i)

            if self.scheduler is not None:
                self.scheduler.step()

    def swad_train_one_domain(self, test_domain):
        train_loader, val_loader, test_loader = self.create_loaders(
            test_domain)
        self.swad_train_regime()
        ind = 0
        val_loss = []
        models = queue.Queue()
      
        while ind < self.swad_config["num_iterations"]:
            for batch in train_loader:
                batch_true, loss = self.process_batch(batch)
                ind += 1
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                accuracy = batch_true.item() / batch["image"].shape[0]

                self.logger.log_metric(
                        f"{self.domains[test_domain]}.train.accuracy", accuracy, ind)
                self.logger.log_metric(
                        f"{self.domains[test_domain]}.train.loss", loss.item(), ind)

                if ind % self.swad_config["frequency"] == 1:
                    averaged_model = AveragedModel(self.model).cpu()
                else:
                    averaged_model.update_parameters(deepcopy(self.model).cpu())

                if ind % self.swad_config["frequency"] == 0:
                    accuracy, loss = self.inference_epoch_model(val_loader)
                    self.swad_train_regime()
                    self.logger.log_metric(
                        f"{self.domains[test_domain]}.val.accuracy", accuracy, ind)
                    self.logger.log_metric(
                        f"{self.domains[test_domain]}.val.loss", loss, ind)
                    val_loss.append(loss)
                    print("IND: ", ind / self.swad_config["frequency"], "LOSS: ", val_loss[-1], "ACCURACY: ", accuracy)
                    if ind == self.swad_config["num_iterations"]:
                        break
                    if self.swad_config["average_finish"]:
                        continue
                    models.put(averaged_model.model)
                    if not self.swad_config["average_begin"]:
                        if models.qsize() <= self.swad_config["n_converge"]:
                            continue
                        if min(val_loss[-self.swad_config["n_converge"]:]) == val_loss[-self.swad_config["n_converge"]]:
                            print("START ITER: ", ind / self.swad_config["frequency"] - self.swad_config["n_converge"] + 1)
                            final_model = AveragedModel(models.get())
                            self.swad_config["average_begin"] = True 
                            loss_threshold = np.mean(val_loss[-self.swad_config["n_converge"]:]) * self.swad_config["tolerance_ratio"]
                            print("THRESHOLD: ", loss_threshold)
                        else:
                            models.get()
                    else:
                        print("CLASSIC model: ", check(self.model, test_loader, nn.CrossEntropyLoss()))
                        self.swad_train_regime()
                        print("SWAD model: ", check(final_model.model.to(self.device), test_loader, nn.CrossEntropyLoss()))
                        print("SWA model: ", check(averaged_model.model.to(self.device), test_loader, nn.CrossEntropyLoss()))
                        
                        
                        if models.qsize() < self.swad_config["n_tolerance"] - 1:
                            continue
                        final_model.update_parameters(models.get())
                        if min(val_loss[-self.swad_config["n_tolerance"]:]) > loss_threshold:
                            print("END ITER: ", ind / self.swad_config["frequency"] - self.swad_config["n_tolerance"] + 1)
                            self.swad_config["average_finish"] = True
                            while models.qsize() > 0:
                                models.get()
                   

        print("Classic model: ", self.inference_epoch_model(test_loader))
        self.model = final_model.model.to(self.device)
        print("SWAD model: ", self.inference_epoch_model(test_loader))
        self.save_checkpoint(test_domain)
        self.swad_config["average_begin"] = False
        self.swad_config["average_finish"] = False

    def train(self):
        # log all info
        all_metrics = defaultdict(list)
        for i in range(len(self.domains)):
            self.init_training()
            if self.swad_config is not None:
                self.swad_train_one_domain(i)
            else:
                self.train_one_domain(i)
            self.load_checkpoint(i)
            metrics = dict()
            for metric, loader in zip(["train", "val", "test"], self.create_loaders(i)):
                metrics[f"{metric}_accuracy"], metrics[f"{metric}_loss"] = self.inference_epoch_model(loader)
            for metric in metrics:
                all_metrics[metric].append(metrics[metric])
        all_metrics["domain"] = self.domains
        self.logger.log_table("all_metrics", pd.DataFrame(all_metrics))

        # log only val and train accuracies
        accuracy_metrics = dict()
        for ind, domain in enumerate(self.domains):
            accuracy_metrics[f"{domain}_val"] = all_metrics["val_accuracy"][ind]
            accuracy_metrics[f"{domain}_test"] = all_metrics["test_accuracy"][ind]
        self.logger.log_table(
            "results", pd.DataFrame(
                accuracy_metrics, index=[1]))

    def save_checkpoint(self, test_domain):
        state = {
            "name": self.config["model"]["name"],
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else [],
            "config": self.config
        }
        path = f"{self.checkpoint_dir}checkpoint_name_{state['name']}_test_domain_{self.domains[test_domain]}_best.pth"
        torch.save(state, path)

    def load_checkpoint(self, test_domain):
        self.model = Parser.init_model(self.model_config, self.device)
        model_path = f"saved/{self.run_id}/checkpoint_name_{self.model_config['name']}"
        model_path += f"_test_domain_{self.domains[test_domain]}_best.pth"
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint["model"])
