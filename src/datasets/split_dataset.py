import random
import os
import numpy as np

all_domains = {
    "PACS": ["art_painting", "cartoon", "photo", "sketch"],
    "VLCS": ["Caltech101", "LabelMe", "SUN09", "VOC2007"],
    "OfficeHome": ["Art", "Clipart", "Product", "Real_World"]
}

all_classes = {
    "PACS": ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"],
    "VLCS": ["bird", "car", "chair", "dog", "person"],
    "OfficeHome": [
        'Alarm_Clock',
        'Backpack',
        'Batteries',
        'Bed',
        'Bike',
        'Bottle',
        'Bucket',
        'Calculator',
        'Calendar',
        'Candles',
        'Chair',
        'Clipboards',
        'Computer',
        'Couch',
        'Curtains',
        'Desk_Lamp',
        'Drill',
        'Eraser',
        'Exit_Sign',
        'Fan',
        'File_Cabinet',
        'Flipflops',
        'Flowers',
        'Folder',
        'Fork',
        'Glasses',
        'Hammer',
        'Helmet',
        'Kettle',
        'Keyboard',
        'Knives',
        'Lamp_Shade',
        'Laptop',
        'Marker',
        'Monitor',
        'Mop',
        'Mouse',
        'Mug',
        'Notebook',
        'Oven',
        'Pan',
        'Paper_Clip',
        'Pen',
        'Pencil',
        'Postit_Notes',
        'Printer',
        'Push_Pin',
        'Radio',
        'Refrigerator',
        'Ruler',
        'Scissors',
        'Screwdriver',
        'Shelf',
        'Sink',
        'Sneakers',
        'Soda',
        'Speaker',
        'Spoon',
        'Table',
        'Telephone',
        'ToothBrush',
        'Toys',
        'Trash_Can',
        'TV',
        'Webcam'
    ]
}

def split_dataset(dataset_name: str, train_size: float = 0.8):
    """Split data to train and test sets for some dataset.
    Uses random, so it is assumed that the seed is already fixed when this function is called.

    Args:
        dataset_name (str): name of dataset.  one of 'PACS', '...'.
        train_size (float, optional): proportion of the training sample. Defaults to 0.8.
    """
    domains = all_domains[dataset_name]
    classes = all_classes[dataset_name]
    for domain in domains:
        for label, cls in enumerate(classes):
            files = np.array(
                sorted(
                    os.listdir(f"data/{dataset_name.lower()}/images/{domain}/{cls}")))
            n = len(files)
            all_indeces = list(np.arange(n))
            if label == 0:
                regime = "w"
            else:
                regime = "a"
            # make train set
            train_indeces = random.sample(all_indeces, k=int(n * train_size))
            with open(f"data/{dataset_name.lower()}/labels/{domain}_train.txt", regime) as f:
                for image in files[train_indeces]:
                    print(
                        f"data/{dataset_name.lower()}/images/{domain}/{cls}/{image} {label}",
                        file=f)
            # make test set
            test_indeces = np.setxor1d(all_indeces, train_indeces)
            with open(f"data/{dataset_name.lower()}/labels/{domain}_test.txt", regime) as f:
                for image in files[test_indeces]:
                    print(
                        f"data/{dataset_name.lower()}/images/{domain}/{cls}/{image} {label}",
                        file=f)
