import torch
import torchvision
import os
from PIL import Image
import random
from src.datasets.base_dataset import BaseDataset

class ImageNet(BaseDataset):
    def __init__(self,
            num_images: int, transforms: torchvision.transforms.Compose,
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
        files = sorted(os.listdir("data/image_net/images/"))
        self.images = random.sample(files, num_images)
        for i in range(num_images):
            self.images[i] = f"data/image_net/images/{self.images[i]}"
        self.labels = torch.zeros(num_images)

    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
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

        image = Image.open(img_name).convert("RGB")

        if self.augmentations:
            sample = {
                "image":
                self.augmentations(image)
            }
        else:
            sample = {
                'image': image,
            }

        sample["image"] = self.transforms(sample["image"])
        sample["label"] = label

        return sample