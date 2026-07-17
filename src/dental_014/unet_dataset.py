import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class MandibleDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_size=(256, 512)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size # (H, W)
        
        # We assume files have same name in both dirs
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, '*.png')))
        self.mask_paths = [os.path.join(mask_dir, os.path.basename(p)) for p in self.image_paths]
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        image = Image.open(img_path).convert("RGB")
        
        # Handle cases where mask might be missing
        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert("L")
        else:
            # Empty mask
            mask = Image.new('L', image.size, 0)
        
        # Resize to fixed size
        image = TF.resize(image, self.img_size)
        mask = TF.resize(mask, self.img_size, interpolation=TF.InterpolationMode.NEAREST)
        
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        
        # Binarize mask
        mask = (mask > 0.5).float()
        
        return image, mask
