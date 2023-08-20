from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm

import pandas as pd
import typing as tp

import torch.utils.data
import torch.nn.functional as F
import torch.nn as nn

from torchvision.utils import make_grid

from src.datasets import create_datasets
from src.parser import Parser
from src.swad import AveragedModel, SWAD
from src.utils import num_iters_loader


class Trainer:
    """Class for training the model in the domain generalization mode.
    """

    def __init__(
            self,
            test_domains,
            config,
            device,
            model_config,
            optimizer_config,
            scheduler_config,
            swad_config,
            dataset,
            num_epochs,
            tracking_step,
            batch_size,
            run_id,
            logger):
        self.test_domains = test_domains
        self.config = config

        self.device = device

        self.model_config = model_config

        self.dataset = dataset

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
        self.checkpoint_dir = f"saved/{self.run_id}/"

    def init_metrics(self):
        self.metrics = defaultdict(lambda: defaultdict(list))
        self.metrics["all_metrics"]["domain"] = self.test_domains
        if self.swad_config is not None:
            self.metrics["all_metrics_erm"]["domain"] = self.test_domains
            if self.swad_config["our_swad_begin"] is not None:
                self.metrics["all_metrics_swad"]["domain"] = self.test_domains

    def init_training(self):
        self.model = Parser.init_model(self.model_config, self.device)
        self.optimizer = Parser.init_optimizer(
            self.optimizer_config, self.model)
        self.scheduler = Parser.init_scheduler(
            self.scheduler_config, self.optimizer)
        if self.swad_config is not None:
            self.swad = SWAD(self.swad_config["n_converge"], self.swad_config["n_tolerance"], self.swad_config["tolerance_ratio"])
        
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
        logits = self.model(images)

        loss = self.loss_function(logits, labels)

        ids = logits.argmax(dim=-1)
        batch_true = (ids == labels).sum()

        return batch_true, loss

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

    def calc_accuracy(self, model, loader):
        with torch.inference_mode():
            model.to(self.device)
            model.eval()
            accuracy = 0
            for batch in loader:
                images, labels = batch["image"], batch["label"]
                images, labels = images.to(self.device).float(), labels.to(self.device).long()
                logits = model(images)
                ids = logits.argmax(dim=-1)
                batch_true = (ids == labels).sum()
                accuracy += batch_true.item()
            model.cpu()
        return accuracy / len(loader.dataset)

    def update_metrics(self, test_domain, name: tp.Optional[str] = None):
        all_name = "all_metrics"
        res_name = "results"
        if name is not None:
            all_name += f"_{name}"
            res_name += f"_{name}"
        for metric, loader in zip(
                ["train", "val", "test"], self.create_loaders(test_domain)):
            accuracy, loss = self.inference_epoch_model(loader)
            self.metrics[all_name][f"{metric}_accuracy"].append(accuracy)
            self.metrics[all_name][f"{metric}_loss"].append(loss)
            if metric != "train":
                self.metrics[res_name][f"{self.test_domains[test_domain]}_{metric}"].append(accuracy)

    def create_loaders(self, test_domain, swad: bool = False):
        train_dataset, val_dataset, test_dataset = create_datasets(
            self.dataset, self.test_domains[test_domain])
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=16)
        if swad:
            train_loader = num_iters_loader(train_loader, self.swad_config["num_iterations"])

        val_loader, test_loader = [torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=16) for dataset in [val_dataset, test_dataset]]
        return train_loader, val_loader, test_loader

    def train_one_domain(self, test_domain):
        train_loader, val_loader, test_loader = self.create_loaders(
            test_domain)
        max_val_accuracy = 0
        self.is_logging = True

        for i in range(1, self.num_epochs + 1):
            train_accuracy, train_loss = self.train_epoch_model(train_loader)
            self.logger.log_metric(
                f"{self.test_domains[test_domain]}.train.accuracy",
                train_accuracy,
                i)
            self.logger.log_metric(
                f"{self.test_domains[test_domain]}.train.loss", train_loss, i)

            val_accuracy, val_loss = self.inference_epoch_model(val_loader)
            self.logger.log_metric(
                f"{self.test_domains[test_domain]}.val.accuracy", val_accuracy, i)
            self.logger.log_metric(
                f"{self.test_domains[test_domain]}.val.loss", val_loss, i)
            if val_accuracy > max_val_accuracy:
                max_val_accuracy = val_accuracy
                self.save_checkpoint(test_domain)

            if i % self.tracking_step == 0:
                test_accuracy, test_loss = self.inference_epoch_model(
                    test_loader)
                self.logger.log_metric(
                    f"{self.test_domains[test_domain]}.test.accuracy", test_accuracy, i)
                self.logger.log_metric(
                    f"{self.test_domains[test_domain]}.test.loss", test_loss, i)

            if self.scheduler is not None:
                self.scheduler.step()

        self.load_checkpoint(test_domain)
        self.update_metrics(test_domain)

    def swad_train_regime(self):
        self.model.train()
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    def swad_train_one_domain(self, test_domain):
        train_loader, val_loader, test_loader = self.create_loaders(test_domain, True)
        max_val_accuracy = 0
        self.is_logging = True
        ind = 0
        self.swad_train_regime()
        
        for batch in train_loader:
            ind += 1
            batch_true, loss = self.process_batch(batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            accuracy = batch_true.item() / batch["image"].shape[0]
            self.logger.log_metric(
                f"{self.test_domains[test_domain]}.train.accuracy", accuracy, ind)
            self.logger.log_metric(
                f"{self.test_domains[test_domain]}.train.loss", loss.item(), ind)

            if self.swad_config["our_swad_begin"] is not None:
                if ind == self.swad_config["our_swad_begin"]:
                    our_swad_model = AveragedModel(self.model)
                    eoa_model = deepcopy(our_swad_model)
                    max_val_eoa_accuracy = self.calc_accuracy(eoa_model.model, val_loader)

                elif ind > self.swad_config["our_swad_begin"]:
                    our_swad_model.update_parameters(deepcopy(self.model).cpu())
                    if ind % self.swad_config["frequency"] == 0:
                        accuracy = self.calc_accuracy(our_swad_model.model, val_loader)
                        if accuracy > max_val_eoa_accuracy:
                            eoa_model = deepcopy(our_swad_model)
                            max_val_eoa_accuracy = accuracy

            if ind % self.swad_config["frequency"] == 1:
                averaged_model = AveragedModel(self.model)
            else:
                averaged_model.update_parameters(
                    deepcopy(self.model).cpu())

            if ind % self.swad_config["frequency"] == 0:
                accuracy, loss = self.inference_epoch_model(val_loader)
                self.swad_train_regime()
                self.swad.update(loss, averaged_model.model)
                self.logger.log_metric(
                    f"{self.test_domains[test_domain]}.val.accuracy", accuracy, ind)
                self.logger.log_metric(
                    f"{self.test_domains[test_domain]}.val.loss", loss, ind)
                if accuracy > max_val_accuracy:
                    max_val_accuracy = accuracy
                    self.save_checkpoint(test_domain)

            if ind % self.tracking_step == 0:
                accuracy, loss = self.inference_epoch_model(test_loader)
                self.swad_train_regime()
                self.logger.log_metric(
                    f"{self.test_domains[test_domain]}.test.accuracy", accuracy, ind)
                self.logger.log_metric(
                    f"{self.test_domains[test_domain]}.test.loss", loss, ind)

        self.load_checkpoint(test_domain)
        self.update_metrics(test_domain, "erm")
        if self.swad_config["our_swad_begin"] is not None:
            self.model = our_swad_model.model.to(self.device)
            self.update_metrics(test_domain)
            self.model = eoa_model.model.to(self.device)
            self.update_metrics(test_domain, "eoa")
        if self.swad.final_model is not None:
            self.swad.finish()
            self.model = self.swad.final_model.model.to(self.device)
            self.save_checkpoint(test_domain)
        if self.swad_config["our_swad_begin"] is not None:
            self.update_metrics(test_domain, "swad")
        else:
            self.update_metrics(test_domain)

    def train(self):
        self.init_metrics()
        for i in range(len(self.test_domains)):
            self.init_training()
            if self.swad_config is not None:
                self.swad_train_one_domain(i)
            else:
                self.train_one_domain(i)
        for name in self.metrics:
            self.logger.log_table(name, pd.DataFrame(self.metrics[name]))

    def save_checkpoint(self, test_domain):
        state = {
            "name": self.config["model"]["name"],
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else [],
            "config": self.config}
        path = f"{self.checkpoint_dir}checkpoint_name_{state['name']}_test_domain_{self.test_domains[test_domain]}_best.pth"
        torch.save(state, path)

    def load_checkpoint(self, test_domain):
        self.model = Parser.init_model(self.model_config, self.device)
        model_path = f"saved/{self.run_id}/checkpoint_name_{self.model_config['name']}"
        model_path += f"_test_domain_{self.test_domains[test_domain]}_best.pth"
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint["model"])
