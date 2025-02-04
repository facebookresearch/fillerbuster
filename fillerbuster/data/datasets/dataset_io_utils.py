import os
import random


def random_traverse_to_target(folder, target: str = "transforms.json"):
    # search the parent folder with random traversal until finding transforms.json in current folder
    # note that this assumes a balanced folder structure!
    if os.path.exists(os.path.join(folder, target)):
        return folder
    if not os.path.isdir(folder):
        return None
    subfolders = os.listdir(folder)
    next_folder = random.choice(subfolders)
    return random_traverse_to_target(os.path.join(folder, next_folder))
