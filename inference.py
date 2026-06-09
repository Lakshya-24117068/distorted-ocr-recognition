import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from utils import VOCABULARY, decode_beam_search
from model import CRNN

def get_sort_key(filename):
    """Sorts filenames like test-123.png numerically."""
    try:
        return int(filename.split('-')[1].split('.')[0])
    except Exception:
        return filename

def preprocess_image(img_path):
    """Loads image and returns three TTA views: original, brightened, contrast-adjusted."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
        
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img)
    
    # Resize to Height=64, Width=256
    img_resized = cv2.resize(img_clahe, (256, 64))
    
    # Normalize to [0, 1]
    img_norm = img_resized.astype(np.float32) / 255.0
    
    # Generate 3 views for TTA:
    # 1. Original
    # 2. Slightly brightened
    # 3. Slightly contrast-adjusted
    view_orig = img_norm
    view_bright = np.clip(img_norm * 1.15, 0.0, 1.0)
    view_contrast = np.clip((img_norm - 0.5) * 1.15 + 0.5, 0.0, 1.0)
    
    # Add channel dimensions
    view_orig = np.expand_dims(view_orig, axis=0)      # (1, 64, 256)
    view_bright = np.expand_dims(view_bright, axis=0)  # (1, 64, 256)
    view_contrast = np.expand_dims(view_contrast, axis=0) # (1, 64, 256)
    
    # Stack into a batch of size 3
    tta_batch = np.stack([view_orig, view_bright, view_contrast], axis=0) # (3, 1, 64, 256)
    return torch.tensor(tta_batch, dtype=torch.float32)

def main():
    test_dir = "cig_ps/test_images"
    checkpoint_path = "best_model.pth"
    output_csv = "submission.csv"
    beam_size = 5
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device for inference: {device}")
    
    # Get sorted test image list
    test_files = [f for f in os.listdir(test_dir) if f.endswith('.png')]
    test_files = sorted(test_files, key=get_sort_key)
    print(f"Found {len(test_files)} test images.")
    
    # Initialize and load model
    num_classes = len(VOCABULARY) + 1
    model = CRNN(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print("Loaded best model checkpoint successfully.")
    
    predictions = []
    
    print("Running inference with TTA + Beam Search...")
    with torch.no_grad():
        for filename in tqdm(test_files):
            img_path = os.path.join(test_dir, filename)
            # Shapes: (3, 1, 64, 256)
            tta_batch = preprocess_image(img_path).to(device)
            
            # Forward pass: logits shape is (64, 3, num_classes)
            logits = model(tta_batch)
            
            # Average logits across the TTA batch dimension (dim 1)
            # Shape becomes (64, num_classes)
            avg_logits = logits.mean(dim=1)
            
            # Compute probabilities
            probs = torch.softmax(avg_logits, dim=-1).cpu().numpy()  # (64, num_classes)
            
            # Beam Search decode
            pred_text = decode_beam_search(probs, beam_size=beam_size)
            predictions.append(pred_text)
            
    # Generate CSV
    df_sub = pd.DataFrame({
        'image': test_files,
        'prediction': predictions
    })
    
    # Verification checks
    print("\nVerifying submission:")
    print(f"  Row count: {len(df_sub)} (Expected: 5000)")
    assert len(df_sub) == 5000, f"Error: Expected 5000 rows, but got {len(df_sub)}"
    
    missing = df_sub['prediction'].isnull().sum()
    empty = (df_sub['prediction'] == "").sum()
    print(f"  Missing values: {missing}")
    print(f"  Empty predictions: {empty}")
    
    # Check for invalid characters
    valid_chars = set(VOCABULARY)
    invalid_rows = 0
    for idx, row in df_sub.iterrows():
        pred = row['prediction']
        if any(c not in valid_chars for c in pred):
            invalid_rows += 1
    print(f"  Rows with invalid characters: {invalid_rows}")
    
    df_sub.to_csv(output_csv, index=False)
    print(f"Submission saved successfully to {output_csv}")

if __name__ == '__main__':
    main()
