import os
import shutil
from sklearn.model_selection import train_test_split
import random


def split_dataset(source_dir, target_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split images from source_dir into train, validation, and test sets in target_dir.

    Args:
        source_dir (str): Path to source folder (e.g., 'all_clash')
        target_dir (str): Path to target folder (e.g., 'split_clash')
        train_ratio (float): Proportion of data for training (default: 0.7)
        val_ratio (float): Proportion of data for validation (default: 0.15)
        test_ratio (float): Proportion of data for testing (default: 0.15)
        seed (int): Random seed for reproducibility
    """
    # Ensure ratios sum to 1
    assert abs(train_ratio + val_ratio + test_ratio -
               1.0) < 1e-5, "Ratios must sum to 1"

    # Create target directories
    splits = ['train', 'validation', 'test']
    for split in splits:
        for cls in os.listdir(source_dir):
            os.makedirs(os.path.join(target_dir, split, cls), exist_ok=True)

    # Set random seed for reproducibility
    random.seed(seed)

    # Iterate over each class (th_10 to th_17)
    for cls in os.listdir(source_dir):
        cls_path = os.path.join(source_dir, cls)
        if not os.path.isdir(cls_path):
            continue

        # Get all image files
        images = [f for f in os.listdir(
            cls_path) if os.path.isfile(os.path.join(cls_path, f))]
        if not images:
            print(f"No images found in {cls_path}")
            continue

        # Shuffle and split images
        images = sorted(images)  # For reproducibility
        random.shuffle(images)

        # Split into train+val and test
        train_val, test = train_test_split(
            images, test_size=test_ratio, random_state=seed)
        # Split train+val into train and val
        train, val = train_test_split(
            train_val, test_size=val_ratio/(train_ratio + val_ratio), random_state=seed)

        # Copy images to respective directories
        for img in train:
            shutil.copy(os.path.join(cls_path, img),
                        os.path.join(target_dir, 'train', cls, img))
        for img in val:
            shutil.copy(os.path.join(cls_path, img), os.path.join(
                target_dir, 'validation', cls, img))
        for img in test:
            shutil.copy(os.path.join(cls_path, img),
                        os.path.join(target_dir, 'test', cls, img))

        print(
            f"Class {cls}: {len(train)} train, {len(val)} val, {len(test)} test images")


# Example usage
source_dir = 'all_clash'
target_dir = 'split_clash'
print('blah')
split_dataset(source_dir, target_dir, train_ratio=0.7,
              val_ratio=0.15, test_ratio=0.15)
