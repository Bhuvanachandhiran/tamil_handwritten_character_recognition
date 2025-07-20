import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TamilDataset(Dataset):
    def __init__(self, csv_file=None, img_dir=None, transform=None, is_test=False):
        self.is_test = is_test
        self.transform = transform
        self.img_dir = Path(img_dir)
        self.data = []
        
        if csv_file:
            self.data = pd.read_csv(csv_file)
            self.data = self.data[self.data['Class Label'] != 156]  # Exclude label 156
            logger.info(f"Loaded {len(self.data)} test images from {csv_file} (label 156 excluded)")
        else:
            for subfolder in self.img_dir.iterdir():
                if subfolder.is_dir():
                    try:
                        class_id = int(subfolder.name)  # Numeric folder names (0 to 155)
                        for img_path in subfolder.glob('*.bmp'):
                            self.data.append({'ID': img_path.name, 'Class Label': class_id})
                    except ValueError:
                        logger.warning(f"Skipping invalid folder: {subfolder.name}")
            if not self.data:
                raise ValueError(f"No valid images found in {self.img_dir}")
            self.data = pd.DataFrame(self.data)
            logger.info(f"Loaded {len(self.data)} training images from {self.img_dir}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]['ID']
        class_id = self.data.iloc[idx]['Class Label']
        # For test data, use Class Label to access subfolder
        img_path = self.img_dir / str(class_id) / img_name if self.is_test else self.img_dir / str(class_id) / img_name
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            logger.warning(f"Image not found: {img_path}")
            return None  # Skip missing images
        image = cv2.resize(image, (32, 32))
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)  # Add channel dimension
        if self.transform:
            image = self.transform(image)
        label = class_id
        return torch.tensor(image), torch.tensor(label, dtype=torch.long)

def custom_collate_fn(batch):
    # Filter out None items (failed image loads)
    batch = [item for item in batch if item is not None]
    if not batch:
        logger.warning("All images in batch failed to load, returning empty batch")
        return None, None  # Return empty batch to skip
    images, labels = zip(*batch)
    return torch.stack(images), torch.stack(labels)

def get_data_loaders(train_dir, test_csv, test_dir, batch_size=32):
    train_dataset = TamilDataset(img_dir=train_dir)
    test_dataset = TamilDataset(csv_file=test_csv, img_dir=test_dir, is_test=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    return train_loader, test_loader