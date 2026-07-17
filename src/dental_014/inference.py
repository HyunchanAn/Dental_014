import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import numpy as np
import os

from dental_014.multitask_model import OsteoMAENet
from dental_014.morphology_analyzer import MorphologyAnalyzer

class OsteoporosisInferencer:
    """
    Inference pipeline for Osteoporosis screening using the new End-to-End Multi-task Architecture.
    """
    def __init__(self, weight_path, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. Load Multi-task Model
        self.model = OsteoMAENet(in_channels=3, out_channels=3, num_classes=3)
        if os.path.exists(weight_path):
            self.model.load_state_dict(torch.load(weight_path, map_location=self.device, weights_only=True))
        else:
            print(f"Warning: Multi-task weights not found at {weight_path}")
            
        self.model.to(self.device)
        self.model.eval()
        
        # 2. Morphology Analyzer (For explainability only)
        self.analyzer = MorphologyAnalyzer(pixels_per_mm=10.0)
        
        self.transform = T.Compose([
            T.Resize((256, 512)),
            T.ToTensor()
        ])
        
        self.classes = ['Normal', 'Osteopenia (경도)', 'Osteoporosis (중증)']
        
    def predict(self, image_path_or_pil):
        """
        Predicts the osteoporosis risk for a given image.
        Returns the class name, the probability distribution, the predicted mask, and morphology features.
        """
        if isinstance(image_path_or_pil, str):
            image = Image.open(image_path_or_pil).convert('RGB')
        else:
            image = image_path_or_pil.convert('RGB')
            
        img_t = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            pred_pixel, class_logits, _ = self.model(img_t, mask=False)
            
            # Reconstructed Image (for visualization instead of mask)
            if pred_pixel is not None:
                recon_img = torch.sigmoid(pred_pixel).squeeze().cpu().numpy()
                if recon_img.ndim == 3:
                    recon_img = np.transpose(recon_img, (1, 2, 0)) # CHW to HWC
                mask = (recon_img * 255).astype(np.uint8) # Return as pseudo-mask/visualization
            else:
                mask = np.zeros((img_t.shape[2], img_t.shape[3], 3), dtype=np.uint8)
            
            # Classification
            probs = F.softmax(class_logits, dim=1).squeeze().cpu().numpy()
            pred_idx = np.argmax(probs)
            
        geom_feats = {"message": "Morphology analysis disabled in Autoencoder (Mask-Free) mode."}
        
        pred_class = self.classes[pred_idx]
        res_dict = {self.classes[i]: float(probs[i]) for i in range(3)}
        
        return pred_class, res_dict, mask, geom_feats
