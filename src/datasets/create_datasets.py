from copy import deepcopy

import src.datasets
from src.utils import init_object


def create_datasets(dataset, test_domain):
    train_dataset, val_dataset, test_dataset = deepcopy(
        dataset), deepcopy(
        dataset), deepcopy(
        dataset)

    train_dataset["kwargs"]["dataset_type"], val_dataset["kwargs"]["dataset_type"] = [
        "train"], ["test"]
    test_dataset["kwargs"]["dataset_type"] = ["train", "test"]

    train_dataset["kwargs"]["domain_list"] = [
        domain for domain in train_dataset["kwargs"]["domain_list"] if domain != test_domain]
    val_dataset["kwargs"]["domain_list"] = [
        domain for domain in val_dataset["kwargs"]["domain_list"] if domain != test_domain]
    test_dataset["kwargs"]["domain_list"] = [test_domain]

    test_dataset["kwargs"]["augmentations"], val_dataset["kwargs"]["augmentations"] = None, None

    train_dataset = init_object(src.datasets, train_dataset)
    val_dataset = init_object(src.datasets, val_dataset)
    test_dataset = init_object(src.datasets, test_dataset)

    return train_dataset, val_dataset, test_dataset
