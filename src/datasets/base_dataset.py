import torch
import torchvision
from PIL import Image


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            transforms: torchvision.transforms.Compose,
            augmentations: torchvision.transforms.Compose = None) -> None:
        self.images = []
        self.labels = torch.Tensor([])
        self.transforms = transforms
        self.augmentations = augmentations


    def __len__(self) -> int:
        """Returns the number of images in the dataset.

        Returns:
            int: dataset len
        """
        return len(self.images)

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
                "image": image,
            }

        sample["image"] = self.transforms(sample["image"])
        sample["label"] = label
        
        return sample
