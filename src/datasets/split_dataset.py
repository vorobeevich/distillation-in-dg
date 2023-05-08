import random
import os
import numpy as np


def split_dataset(dataset_name: str, train_size: float = 0.8):
    """Split data to train and test sets for some dataset.
    Uses random, so it is assumed that the seed is already fixed when this function is called.

    Args:
        dataset_name (str): name of dataset.  one of 'PACS', '...'.
        train_size (float, optional): proportion of the training sample. Defaults to 0.8.
    """
    if dataset_name == "PACS":
        split_pacs(train_size)


def split_pacs(train_size: float):
    domains = ["art_painting", "cartoon", "photo", "sketch"]
    classes = [
        "dog",
        "elephant",
        "giraffe",
        "guitar",
        "horse",
        "house",
        "person"
    ]
    for domain in domains:
        for label, cls in enumerate(classes):
            files = np.array(
                sorted(
                    os.listdir(f"data/pacs/images/{domain}/{cls}")))
            n = len(files)
            all_indeces = list(np.arange(n))
            if label == 0:
                regime = "w"
            else:
                regime = "a"
            # make train set
            train_indeces = random.sample(all_indeces, k=int(n * train_size))
            with open(f"data/pacs/labels/{domain}_train.txt", regime) as f:
                for image in files[train_indeces]:
                    print(
                        f"data/pacs/images/{domain}/{cls}/{image} {label}",
                        file=f)
            # make test set
            test_indeces = np.setxor1d(all_indeces, train_indeces)
            with open(f"data/pacs/labels/{domain}_test.txt", regime) as f:
                for image in files[test_indeces]:
                    print(
                        f"data/pacs/images/{domain}/{cls}/{image} {label}",
                        file=f)
