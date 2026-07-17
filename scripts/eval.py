import os
import sys
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from dental_014.dataset import OsteoporosisDataset
from torch.utils.data import DataLoader
from dental_014.multitask_model import OsteoMultiTaskNet

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    test_dir = os.path.join("data", "clean_cropped_ds", "test")
    val_dataset = OsteoporosisDataset(test_dir, split='test')
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    model = OsteoMultiTaskNet(in_channels=3, out_channels=3, num_classes=3)
    weight_path = os.path.join("weights", "best.pt")
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    print("Evaluating...")
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            recon_logits, outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    classes = ['Normal(C1)', 'Osteopenia(C2)', 'Osteoporosis(C3)']
    
    print("\n--- Classification Report ---")
    print(classification_report(all_labels, all_preds, target_names=classes, zero_division=0))
    
    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)

if __name__ == "__main__":
    evaluate()
