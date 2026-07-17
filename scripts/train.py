import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from dental_014.dataset import OsteoporosisDataset
from dental_014.model import get_model

def parse_args():
    parser = argparse.ArgumentParser(description="Train Osteoporosis Screening Model")
    parser.add_argument("--data_dir", type=str, default="data", help="Root data directory")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--save_dir", type=str, default="weights", help="Directory to save weights")
    return parser.parse_args()

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Paths for train and test splits (새로운 크롭 원본 데이터셋용 경로)
    train_dir = os.path.join(args.data_dir, "clean_cropped_ds", "train")
    test_dir = os.path.join(args.data_dir, "clean_cropped_ds", "test")
    
    # Dataset and DataLoader
    train_dataset = OsteoporosisDataset(train_dir, split='train')
    val_dataset = OsteoporosisDataset(test_dir, split='test')
    
    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Class weights for imbalanced dataset (example values, could be calculated dynamically)
    # C1: ~10606, C2: ~7381, C3: ~2310
    total = 10606 + 7381 + 2310
    weights = [total/10606, total/7381, total/2310]
    class_weights = torch.FloatTensor(weights).to(device)
    
    model = get_model(num_classes=3, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels.data)
            
            pbar.set_postfix({'loss': loss.item()})
            
        train_loss = train_loss / len(train_dataset)
        train_acc = train_correct.double() / len(train_dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data)
                
        val_loss = val_loss / len(val_dataset)
        val_acc = val_correct.double() / len(val_dataset)
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{args.epochs} - "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best.pt'))
            print("Saved best model.")

if __name__ == "__main__":
    args = parse_args()
    train(args)
