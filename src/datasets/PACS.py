import torch
import torchvision
from src.datasets.base_dataset import BaseDataset


class PACS(BaseDataset):
    """Class with standard methods for torch dataset for working with PACS dataset.
    Inherited from standard class torch.utils.data.Dataset.
    Dataset paper: https://arxiv.org/abs/1710.03077.
    """

    def __init__(
            self,
            dataset_type: list[str],
            domain_list: list[str],
            transforms: torchvision.transforms.Compose,
            augmentations: torchvision.transforms.Compose = None) -> None:
        """Dataset initialization. Creates images list (where file paths are stored)
        and classes (labels) torch.Tensor for them.

        Args:
            dataset_types (list[str]): list of values from {'train', 'test'}.
            domain (list[str]): list of values from {'art_painting', 'cartoon', 'photo', 'sketch'}.
            transforms (torchvision.transforms.Compose): transforms that are applied to each
                image (regardless of whether it is in train or test selection).
            augmentations (torchvision.transforms.Compose,
            optional): augmentations that apply only to the train selection. Defaults to None.
        """
        super().__init__(transforms, augmentations)
        self.domain_list = domain_list
        for domain in domain_list:
            imgs, lbls = self.get_paths_and_labels(dataset_type, domain)
            self.images += imgs
            self.labels = torch.cat((self.labels, lbls))

    def get_paths_and_labels(self,
                             dataset_types: list[str],
                             domain: str) -> tuple[list[str],
                                                   torch.Tensor]:
        """Return list of images paths for a given type of the dataset.

        Args:
            dataset_types (list[str]): list of values from {'train', 'test'}.
            domain (str): one of 'art_painting', 'cartoon', 'photo', 'sketch'.

        Returns:
            tuple[list[str], torch.Tensor]: paths to images and tensor with class labels.
        """

        paths = []
        labels = []
        for ds_type in dataset_types:
            filepath = f"data/pacs/labels/{domain}_{ds_type}.txt"
            f = open(filepath, 'r')
            lines = f.readlines()
            f.close()
            lines = [l.split() for l in lines]
            cur_paths, cur_labels = zip(*lines)
            cur_labels = [int(l) for l in cur_labels]
            paths += cur_paths
            labels += cur_labels
        return paths, torch.Tensor(labels)