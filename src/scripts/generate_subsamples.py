import argparse
import os
import random
import shutil
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Train model from config")

parser.add_argument(
    "--data_path",
    type=str,
    help="Path to folder with subfolders with images",
    required=True
)
parser.add_argument(
    "--save_path",
    type=str,
    help="Path to folder for saving subsample",
    required=True
)
parser.add_argument(
    "--number",
    type=int,
    help="Number of images for result folder",
    required=True
)

args = parser.parse_args()

# make dir for saving images
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

all_folders = os.listdir(args.data_path)
for i in tqdm(range(args.number)):
    # select random folder
    folder = f"{args.data_path}/{random.sample(all_folders, 1)[0]}"
    all_files = os.listdir(folder)
    # select random image and copy it
    shutil.copy(f"{folder}/{random.sample(all_files, 1)[0]}", f"{args.save_path}/{i}.jpg")
