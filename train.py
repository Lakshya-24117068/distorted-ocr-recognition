import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from utils import VOCABULARY, decode_greedy, calculate_cer_batch
from dataset import get_ocr_dataloaders
from model import CRNN

def set_seed(seed=42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_sanity_check(model, device):
    """Runs a sanity check forward pass on the model."""
    print("=================================================")
    print("PHASE 11: SANITY CHECK BEFORE TRAINING")
    print("=================================================")
    
    # Create a dummy batch: Batch=4, Channels=1, Height=64, Width=256
    dummy_input = torch.randn(4, 1, 64, 256, device=device)
    
    # Model forward pass
    model.eval()
    with torch.no_grad():
        logits = model(dummy_input)
        
    print(f"Input shape:                {dummy_input.shape}")
    print(f"Output logits shape:        {logits.shape} (SeqLen, Batch, VocabSize + 1)")
    
    seq_len, batch_size, num_classes = logits.shape
    
    # Verify outputs
    assert seq_len == 64, f"Expected Sequence Length of 64, but got {seq_len}"
    assert batch_size == 4, f"Expected Batch Size of 4, but got {batch_size}"
    assert num_classes == len(VOCABULARY) + 1, f"Expected {len(VOCABULARY) + 1} classes, but got {num_classes}"
    print("[OK] Tensor shapes verified successfully.")
    
    # Verify CTC Loss compatibility
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    log_probs = logits.log_softmax(2)
    input_lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=device)
    targets = torch.tensor([1, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 7, 3, 4, 5, 6, 7, 8, 4, 5, 6, 7, 8, 9], dtype=torch.long, device=device)
    target_lengths = torch.tensor([6, 6, 6, 6], dtype=torch.long, device=device)
    
    loss = criterion(log_probs, targets, input_lengths, target_lengths)
    print(f"Mock CTC Loss computed:     {loss.item():.4f}")
    assert not torch.isnan(loss) and not torch.isinf(loss), "CTC Loss returned nan/inf!"
    print("[OK] CTC compatibility verified successfully.")
    
    # Verify decoder correctness
    preds = log_probs.argmax(2).permute(1, 0).cpu().numpy()  # (Batch, SeqLen)
    decoded = [decode_greedy(pred) for pred in preds]
    print(f"Greedy decoded sample:      '{decoded[0]}'")
    print("[OK] Decoder correctness verified successfully.")
    print("Sanity check passed! Ready for training.\n")
    model.train()


def train_epoch(model, dataloader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    
    for batch_idx, (images, targets, target_lengths, _) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        target_lengths = target_lengths.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        # Mixed precision forward pass
        with autocast():
            logits = model(images)
            log_probs = logits.log_softmax(2)
            seq_len = logits.size(0)
            batch_size = logits.size(1)
            
            input_lengths = torch.full(
                size=(batch_size,),
                fill_value=seq_len,
                dtype=torch.long,
                device=device
            )
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            
        # Scaling and backpropagation
        scaler.scale(loss).backward()
        
        # Unscale for gradient clipping
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        # Step optimizer and scaler
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item() * batch_size
        
    return total_loss / len(dataloader.dataset)


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets, target_lengths, labels in dataloader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            target_lengths = target_lengths.to(device, non_blocking=True)
            
            with autocast():
                logits = model(images)
                log_probs = logits.log_softmax(2)
                seq_len = logits.size(0)
                batch_size = logits.size(1)
                
                input_lengths = torch.full(
                    size=(batch_size,),
                    fill_value=seq_len,
                    dtype=torch.long,
                    device=device
                )
                loss = criterion(log_probs, targets, input_lengths, target_lengths)
                
            total_loss += loss.item() * batch_size
            
            # Greedy decoding for validation CER
            preds = log_probs.argmax(2).permute(1, 0).cpu().numpy()  # (Batch, SeqLen)
            for pred in preds:
                all_preds.append(decode_greedy(pred))
            all_targets.extend(labels)
            
    val_loss = total_loss / len(dataloader.dataset)
    val_cer = calculate_cer_batch(all_preds, all_targets)
    
    return val_loss, val_cer, all_preds, all_targets


def main():
    set_seed(42)
    
    # Paths
    csv_path = "cig_ps/train-labels.csv"
    img_dir = "cig_ps/train_images"
    
    # Hyperparameters
    batch_size = 64
    max_epochs = 60
    learning_rate = 1e-3
    weight_decay = 1e-4
    early_stopping_patience = 10
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset splits
    train_loader, val_loader = get_ocr_dataloaders(
        csv_path=csv_path,
        img_dir=img_dir,
        batch_size=batch_size,
        val_ratio=0.15,
        seed=42
    )
    
    # Initialize model
    num_classes = len(VOCABULARY) + 1  # 31 standard + 1 blank
    model = CRNN(num_classes=num_classes).to(device)
    
    # Run pre-training sanity check
    run_sanity_check(model, device)
    
    # Loss, Optimizer, Scaler, and Scheduler
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scaler = GradScaler()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        verbose=True
    )
    
    best_val_cer = float('inf')
    best_epoch = 0
    epochs_no_improve = 0
    
    print("Starting training loop...")
    print(f"{'Epoch':<6} | {'Train Loss':<10} | {'Val Loss':<8} | {'Val CER':<7} | {'Time (s)':<8}")
    print("-" * 50)
    
    for epoch in range(1, max_epochs + 1):
        start_time = time.time()
        
        # Train & Validate
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_cer, val_preds, val_targets = validate(model, val_loader, criterion, device)
        
        elapsed = time.time() - start_time
        
        # LR step based on val CER
        scheduler.step(val_cer)
        
        print(f"{epoch:<6} | {train_loss:<10.4f} | {val_loss:<8.4f} | {val_cer:<7.4f} | {elapsed:<8.1f}")
        
        # Every 5 epochs, display 5 predictions alongside ground truth labels
        if epoch % 5 == 0 or epoch == 1:
            print("\n--- Validation Samples (Epoch {}) ---".format(epoch))
            indices = random.sample(range(len(val_preds)), min(5, len(val_preds)))
            for idx in indices:
                print(f"  GT: '{val_targets[idx]}' | Pred: '{val_preds[idx]}'")
            print("-" * 38 + "\n")
            
        # Check and save best model
        if val_cer < best_val_cer:
            best_val_cer = val_cer
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_model.pth")
            print(f"** Saved new best model checkpoint (Val CER: {val_cer:.4f}) **")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print(f"\nEarly stopping triggered. No validation CER improvement for {early_stopping_patience} epochs.")
                break
                
    print("\nTraining completed.")
    print(f"Best Epoch: {best_epoch} | Lowest Validation CER: {best_val_cer:.4f}")
    
if __name__ == '__main__':
    main()
