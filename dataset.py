import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A

from utils import encode_text

class OCRDataset(Dataset):
    def __init__(self, df, img_dir, is_train=True):
        """
        df: pandas DataFrame with columns ['image', 'text']
        img_dir: path to directory containing images
        is_train: boolean, whether this is training mode (applies augmentations)
        """
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.is_train = is_train
        
       
        
        # Lightweight training augmentations
        self.transform = A.Compose([
            A.Rotate(limit=10, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=255),
            A.Perspective(scale=(0.01, 0.05), p=0.5, pad_val=255),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=0, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=255),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.5),
            A.CoarseDropout(max_holes=4, max_height=8, max_width=8, min_holes=1, min_height=2, min_width=2, fill_value=255, p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5)
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['image']
        label_text = row['text']
        
        img_path = os.path.join(self.img_dir, img_name)
        # 1. Load image as grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
            
        # 2. Apply CLAHE contrast enhancement
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
            )

        img = clahe.apply(img)
        
        
        # 3. Apply augmentations (Training only)
        if self.is_train:
            augmented = self.transform(image=img)
            img = augmented['image']
            
        # 4. Resize to target dimension Height=64, Width=256
        img = cv2.resize(img, (256, 64))
        
        # 5. Normalization
        img = img.astype(np.float32) / 255.0
        
        # 6. Convert to PyTorch Tensor shape (1, 64, 256)
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        
        # 7. Encode target sequence
        target_encoded = encode_text(label_text)
        target_tensor = torch.tensor(target_encoded, dtype=torch.long)
        
        return img_tensor, target_tensor, label_text


def get_ocr_dataloaders(csv_path, img_dir, batch_size=64, val_ratio=0.15, seed=42):
    """
    Reads the CSV, filters corrupted records, splits into train/validation sets,
    and returns PyTorch Dataloaders.
    """
    df = pd.read_csv(csv_path)
    
    # Standardize columns (csv columns are ['', 'image', 'text'] or similar)
    if 'text' not in df.columns and 'label' in df.columns:
        df = df.rename(columns={'label': 'text'})
    elif 'text' not in df.columns and len(df.columns) >= 3:
        df.columns = ['id', 'image', 'text']
        
    df = df[['image', 'text']].copy()
    df['text'] = df['text'].astype(str)
    
    # Identify and remove the 2 corrupted labels
    corrupted_images = {"train-2184.png", "train-6819.png"}
    df_clean = df[~df['image'].isin(corrupted_images)].copy()
    
    # Reproducible split
    df_shuffled = df_clean.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_size = int(len(df_shuffled) * val_ratio)
    
    df_val = df_shuffled.iloc[:val_size]
    df_train = df_shuffled.iloc[val_size:]
    
    print(f"Dataset split summary:")
    print(f"  Total clean samples: {len(df_shuffled)}")
    print(f"  Training samples:    {len(df_train)}")
    print(f"  Validation samples:  {len(df_val)}")
    
    train_dataset = OCRDataset(df_train, img_dir, is_train=True)
    val_dataset = OCRDataset(df_val, img_dir, is_train=False)
    
    # Collate function to handle variable target lengths (CTC standard)
    def collate_fn(batch):
        images, targets, labels = zip(*batch)
        
        images = torch.stack(images, dim=0)
        
        # Concatenate targets for CTCLoss and keep track of lengths
        target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
        targets_concat = torch.cat(targets)
        
        return images, targets_concat, target_lengths, labels

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, val_loader
