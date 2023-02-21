import os
from copy import deepcopy
from tqdm import tqdm

import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models

import src.datasets.PACS_dataset
from src.utils.init_functions import init_object

class Trainer:

    def __init__(self, config, device, model, optimizer, scheduler, dataset, num_epochs, batch_size, run_id, logger):
        self.config = config

        self.device = device
        
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.dataset = dataset
        self.domains = dataset["kwargs"]["domain_list"]

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        
        self.logger = logger
        
        self.run_id = run_id
        self.checkpoint_dir = "saved/"
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.checkpoint_dir += f"{self.run_id}/"
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

    def train_epoch_model(self, loader, model, loss_function, optimizer):
        model.train()
        accuracy = 0 
        loss_sum = 0
        pbar = tqdm(loader, leave=True)
        for batch in pbar:
            images, labels = batch["image"], batch["label"]
            images, labels = images.to(self.device).float(), labels.to(self.device).long()
            logits = model(images)
            loss = loss_function(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * images.shape[0]
            
            ids = F.softmax(logits, dim=-1).argmax(dim=-1)
            batch_true = (ids == labels).sum() 
            accuracy += batch_true.item()
            
            pbar.set_description("Accuracy on batch %f loss on batch %f" % ((batch_true / images.shape[0]).item(), loss.item()))

        return accuracy / len(loader.dataset), loss_sum  / len(loader.dataset)

    def inference_epoch_model(self, loader, model, loss_function):
        with torch.inference_mode():
            model.eval()
            accuracy = 0 
            loss_sum = 0
            pbar = tqdm(loader, leave=True)
            for batch in pbar:
                images, labels = batch["image"].to(self.device), batch["label"].to(self.device).long()
                logits = model(images)
                loss = loss_function(logits, labels)
                loss_sum += loss.item() * images.shape[0]

                ids = F.softmax(logits, dim=-1).argmax(dim=-1)
                batch_true = (ids == labels).sum() 
                accuracy += batch_true.item()

                pbar.set_description("Accuracy on batch %f loss on batch %f" % ((batch_true / images.shape[0]).item(), loss.item()))

            return accuracy / len(loader.dataset), loss_sum / len(loader.dataset) 
    
    def train_model(self, test_domain, model, loss_function, optimizer, scheduler):
        train_loader, val_loader, test_loader = self.create_loaders(test_domain)
        max_val_accuracy = 0
        for i in range(1, self.num_epochs + 1):
            train_accuracy, train_loss = self.train_epoch_model(train_loader, model, loss_function, optimizer) 
            self.logger.log_epoch(self.domains[test_domain], 'train', 'accuracy', train_accuracy, i)
            self.logger.log_epoch(self.domains[test_domain], 'train', 'loss', train_loss, i)
            
            val_accuracy, val_loss = self.inference_epoch_model(val_loader, model, loss_function)
            self.logger.log_epoch(self.domains[test_domain], 'val', 'accuracy', val_accuracy, i)
            self.logger.log_epoch(self.domains[test_domain], 'val', 'loss', val_loss, i)
            if val_accuracy > max_val_accuracy:
                max_val_accuracy = val_accuracy
                self.save_checkpoint(test_domain, model, optimizer, scheduler)

            test_accuracy, test_loss = self.inference_epoch_model(test_loader, model, loss_function)
            self.logger.log_epoch(self.domains[test_domain], 'test', 'accuracy', test_accuracy, i)
            self.logger.log_epoch(self.domains[test_domain], 'test', 'loss', test_loss, i, True)
            scheduler.step()

    def train(self):       
        for i in range(len(self.domains)):
            # init model params
            model = init_object(torchvision.models, self.config["model"])
            model.fc = nn.Linear(*self.config["last_layer"])
            
            # prepare for GPU training
            model.to(self.device)
            
            # init optimizer with model params
            optimizer = deepcopy(self.config["optimizer"])
            optimizer["kwargs"].update(params=model.parameters())
            optimizer = init_object(torch.optim, optimizer)
            
            # init scheduler with optimizer params
            scheduler = deepcopy(self.config["scheduler"])
            scheduler["kwargs"].update(optimizer=optimizer)
            scheduler = init_object(torch.optim.lr_scheduler, scheduler)
            self.train_model(i, model, nn.CrossEntropyLoss(),  optimizer, scheduler)

    def save_checkpoint(self, test_domain, model, optimizer, scheduler):
        state = {
            "name": self.config["model"]["name"],
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": self.config
        }
        path = f"{self.checkpoint_dir}checkpoint_name_{state['model']}_test_domain_{test_domain}_best.pth"
        torch.save(state, path)