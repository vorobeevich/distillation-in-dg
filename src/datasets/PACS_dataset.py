import torch
import torchvision
from PIL import Image


def get_paths_and_labels(
        dataset_types: list[str], domain: str) -> tuple[list[str], torch.Tensor]:
    """Return list of images paths for a given type of the dataset.

    Args:
        dataset_types (list[str]): list of values from {'train', 'test'}
        domain (str): one of 'art_painting', 'cartoon', 'photo', 'sketch'

    Returns:
        tuple[list[str], torch.Tensor]: paths to images and tensor with class labels
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


class PACS_dataset(torch.utils.data.Dataset):
    """Class with standard methods for torch dataset for working with PACS dataset.
    Inherited from standard class torch.utils.data.Dataset.
    """

    def init(
            self,
            dataset_type: list[str],
            domain_list: list[str],
            transforms: torchvision.transforms.Compose,
            augmentations: torchvision.transforms.Compose = None) -> None:
        """Dataset initialization. Creates images list (where file paths are stored) 
        and classes (labels) torch.Tensor for them.

        Args:
            dataset_types (list[str]): list of values from {'train', 'test'}
            domain (list[str]): list of values from {'art_painting', 'cartoon', 'photo', 'sketch'}
            transforms (torchvision.transforms.Compose): transforms that are applied to each
                image (regardless of whether it is in train or test selection)
            augmentations (torchvision.transforms.Compose,
            optional): augmentations that apply only to the train selection. Defaults to None.
        """
        self.images = []
        self.labels = torch.Tensor([])
        self.domain_list = domain_list
        for domain in domain_list:
            imgs, lbls = get_paths_and_labels(dataset_type, domain)
            self.images += imgs
            self.labels = torch.cat((self.labels, lbls))

        self.transforms = transforms
        self.augmentations = augmentations

    def len(self) -> int:
        """Returns the number of images in the dataset.

        Returns:
            int: dataset len
        """
        return len(self.images)

    def getitem(self, idx: int) -> dict[str, torch.Tensor]:
        """Returns a picture from the dataset by its number.
        First, the image is read along the path, augmentations are applied to it (if necessary), then transforms.
        Also, the class label is returned.

        Args:
            idx (int): index of image
        Returns:
            dict[str, torch.Tensor]: dict:
                {
                    "image": image torch.Tensor,
                    "label": label torch.Tensor
                }
        """
        img_name = self.images[idx]
        label = self.labels[idx]

        image = Image.open(img_name)

        if self.augmentations:
            sample = {
                'image':
                self.augmentations(image)
            }
        else:
            sample = {
                'image': image,
            }

        sample['image'] = self.transforms(sample['image'])
        sample['label'] = label

        return sample