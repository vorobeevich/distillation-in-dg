import torch
from PIL import Image

def get_paths_and_labels(dataset_types: list[str], domain: str) -> tuple[list[str], torch.Tensor]:
    """Return list of images paths for a given type of the dataset.

    Args:
        dataset_types (list[str]): one of ['train'], ['test'], ['train', 'test']
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
    labels = torch.Tensor(labels) - 1
    return paths, labels

class PACS_dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_type, domain_list, transforms, augmentations=None):
        self.images = []
        self.labels = torch.Tensor([])
        self.domain_list = domain_list
        for domain in domain_list:
            imgs, lbls = get_paths_and_labels(dataset_type, domain)
            self.images += imgs
            self.labels = torch.cat((self.labels, lbls))
        
        self.transforms = transforms
        self.augmentations = augmentations

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):        
        img_name = self.images[idx]
        label = self.labels[idx]
        
        image = Image.open(f"data/pacs/images/{img_name}") 

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