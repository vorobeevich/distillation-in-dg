import sys
sys.path.append("./")

import random
import os
import numpy as np

from src.utils.fix_seed import fix_seed

fix_seed()

domains = ["art_painting", "cartoon", "photo", "sketch"]
classes = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]
for domain in domains:
    for label, cls in enumerate(classes):
        files = np.array(sorted(os.listdir(f"data/pacs/images/{domain}/{cls}")))
        n = len(files)
        all_indeces = list(np.arange(n))
        # make train set
        train_size = int(n * 0.8)
        train_indeces = random.sample(all_indeces, k=train_size)
        with open(f"data/pacs/labels/{domain}_train.txt", "a") as f:
            for image in files[train_indeces]:
                print(f"data/pacs/images/{domain}/{cls}/{image} {label}", file=f)
        # make test set
        test_size = n - train_size
        test_indeces = np.setxor1d(all_indeces, train_indeces)
        with open(f"data/pacs/labels/{domain}_test.txt", "a") as f:
            for image in files[test_indeces]:
                print(f"data/pacs/images/{domain}/{cls}/{image} {label}", file=f)
