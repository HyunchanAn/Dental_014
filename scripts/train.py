import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from dental_014.dataset import OsteoporosisDataset
from dental_014.multitask_model import OsteoMAENet
from dental_014.loss import OsteoCompositeLoss

def parse_args():
    parser = argparse.ArgumentParser(description="Train OsteoMAENet Model")
    parser.add_argument("--data_dir", type=str, default="data", help="Root data directory")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save_dir", type=str, default="weights", help="Directory to save weights")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from")
    return parser.parse_args()

def patchify(imgs, patch_size=16):
    """
    imgs: (N, 3, H, W)
    x: (N, L, patch_size**2 *3)
    """
    p = patch_size
    assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0

    h = imgs.shape[2] // p
    w = imgs.shape[3] // p
    x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
    return x

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, 'logs'))
    
    train_dir = os.path.join(args.data_dir, "clean_cropped_ds", "train")
    test_dir = os.path.join(args.data_dir, "clean_cropped_ds", "test")
    
    train_dataset = OsteoporosisDataset(train_dir, split='train')
    val_dataset = OsteoporosisDataset(test_dir, split='test')
    
    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    model = OsteoMAENet()
    if args.resume and os.path.exists(args.resume):
        model.load_state_dict(torch.load(args.resume, map_location='cpu'))
        print(f"Resumed weights from {args.resume}")
    model = model.to(device)
    
    criterion = OsteoCompositeLoss(recon_weight=1.0, ordinal_weight=1.0, supcon_weight=0.5).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            pred_pixel, class_logits, supcon_embeds = model(inputs, mask=True)
            target_pixel = patchify(inputs)
            
            loss, loss_recon, loss_ordinal, loss_supcon = criterion(pred_pixel, target_pixel, class_logits, supcon_embeds, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(class_logits, 1)
            train_correct += torch.sum(preds == labels.data)
            
            pbar.set_postfix({'loss': loss.item()})
            
        train_loss = train_loss / len(train_dataset)
        train_acc = train_correct.double() / len(train_dataset)
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Validation uses mask=False
                pred_pixel, class_logits, supcon_embeds = model(inputs, mask=False)
                
                loss, loss_recon, loss_ordinal, loss_supcon = criterion(None, None, class_logits, supcon_embeds, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(class_logits, 1)
                val_correct += torch.sum(preds == labels.data)
                
        val_loss = val_loss / len(val_dataset)
        val_acc = val_correct.double() / len(val_dataset)
        
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best.pt'))
            print("Saved best model.")
            
    writer.close()

if __name__ == "__main__":
    args = parse_args()
    train(args)
