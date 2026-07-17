import os
import sys
import torch
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from dental_014.unet_model import UNet
from dental_014.morphology_analyzer import MorphologyAnalyzer
import torchvision.transforms as T

def extract_features(img_dir, unet_model, analyzer, device):
    classes = ['C1', 'C2', 'C3']
    features = []
    labels = []
    
    transform = T.Compose([
        T.Resize((256, 512)),
        T.ToTensor()
    ])
    
    class_map = {'C1': 0, 'C2': 1, 'C3': 2}
    for fname in tqdm(os.listdir(img_dir), desc=f"Extracting {img_dir}"):
        ext = fname.lower()
        if not (ext.endswith('.png') or ext.endswith('.jpg') or ext.endswith('.jpeg')):
            continue
            
        cls_prefix = fname.split('_')[0]
        if cls_prefix not in class_map:
            continue
        cls_idx = class_map[cls_prefix]
        
        img_path = os.path.join(img_dir, fname)
        
        img = Image.open(img_path).convert('RGB')
        img_t = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            out = unet_model(img_t)
            # sigmoid and threshold
            mask = torch.sigmoid(out).squeeze().cpu().numpy()
            mask = (mask > 0.5).astype(np.uint8) * 255
            
        geom_feats = analyzer.analyze_mask(mask)
        
        feat_vector = [
            geom_feats['mean_thickness_mm'],
            geom_feats['std_thickness_mm'],
            geom_feats['min_thickness_mm'],
            geom_feats['porosity_index'],
            geom_feats['solidity']
        ]
        features.append(feat_vector)
        labels.append(cls_idx)
            
    return np.array(features), np.array(labels)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_dir = os.path.join(base_dir, 'data', 'clean_cropped_ds', 'train')
    test_dir = os.path.join(base_dir, 'data', 'clean_cropped_ds', 'test')
    
    # Load U-Net
    unet = UNet(in_channels=3, out_channels=1).to(device)
    unet_path = os.path.join(base_dir, 'weights', 'unet_mandible_best.pth')
    unet.load_state_dict(torch.load(unet_path, map_location=device, weights_only=True))
    unet.eval()
    
    analyzer = MorphologyAnalyzer(pixels_per_mm=10.0)
    
    print("Extracting training features...")
    X_train, y_train = extract_features(train_dir, unet, analyzer, device)
    
    print("Extracting testing features...")
    X_test, y_test = extract_features(test_dir, unet, analyzer, device)
    
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf.fit(X_train, y_train)
    
    print("Evaluating...")
    preds = rf.predict(X_test)
    print(classification_report(y_test, preds, target_names=['Normal(C1)', 'Osteopenia(C2)', 'Osteoporosis(C3)']))
    
    rf_path = os.path.join(base_dir, 'weights', 'rf_classifier.pkl')
    with open(rf_path, 'wb') as f:
        pickle.dump(rf, f)
    print(f"Saved {rf_path}!")

if __name__ == "__main__":
    main()
