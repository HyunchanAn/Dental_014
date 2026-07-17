import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from dental_014.model import get_model

class OsteoporosisInferencer:
    """
    Inference pipeline for Osteoporosis screening.
    """
    def __init__(self, weight_path, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = get_model(num_classes=3, pretrained=False)
        self.model.load_state_dict(torch.load(weight_path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        
        self.classes = ['Normal', 'Osteopenia (경도)', 'Osteoporosis (중증)']
        
    def predict(self, image_path_or_pil):
        """
        Predicts the osteoporosis risk for a given image.
        Returns the class name and the probability distribution.
        """
        if isinstance(image_path_or_pil, str):
            image = Image.open(image_path_or_pil).convert('RGB')
        else:
            image = image_path_or_pil.convert('RGB')
            
        img_t = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(img_t)
            probs = F.softmax(outputs, dim=1).squeeze().cpu().numpy()
            
        pred_idx = probs.argmax()
        pred_class = self.classes[pred_idx]
        
        res_dict = {self.classes[i]: float(probs[i]) for i in range(3)}
        
        return pred_class, res_dict
