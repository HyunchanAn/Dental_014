import os
import glob
import torch
from PIL import Image
from tqdm import tqdm
import torchvision.transforms.functional as TF
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from dental_014.unet_model import UNet

def generate_masks(data_dir, split, model, device):
    img_dir = os.path.join(data_dir, "clean_cropped_ds", split)
    mask_dir = os.path.join(data_dir, "clean_cropped_ds", f"{split}_masks")
    os.makedirs(mask_dir, exist_ok=True)
    
    img_paths = glob.glob(os.path.join(img_dir, '*.png')) + glob.glob(os.path.join(img_dir, '*.jpg'))
    print(f"Generating masks for {split} split: {len(img_paths)} images")
    
    for img_path in tqdm(img_paths):
        filename = os.path.basename(img_path)
        mask_filename = os.path.splitext(filename)[0] + ".png"
        mask_path = os.path.join(mask_dir, mask_filename)
        
        if os.path.exists(mask_path):
            continue
            
        image = Image.open(img_path).convert("RGB")
        orig_size = image.size
        
        image = TF.resize(image, (256, 512))
        img_tensor = TF.to_tensor(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(img_tensor)
            mask = torch.sigmoid(output).squeeze().cpu().numpy()
            
        mask = (mask > 0.5).astype("uint8") * 255
        mask_img = Image.fromarray(mask, mode="L")
        mask_img = mask_img.resize(orig_size, Image.Resampling.NEAREST)
        mask_img.save(mask_path)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model_path = "weights/unet_mandible_best.pth"
    if not os.path.exists(model_path):
        print(f"Error: Could not find U-Net weights at {model_path}")
        sys.exit(1)
        
    model = UNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    generate_masks("data", "train", model, device)
    generate_masks("data", "test", model, device)
    print("Mask generation complete!")
