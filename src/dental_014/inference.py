import os
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import numpy as np

try:
    from .multitask_model import OsteoMAENet
    from .morphology_analyzer import MorphologyAnalyzer
except ImportError:
    from dental_014.multitask_model import OsteoMAENet
    from dental_014.morphology_analyzer import MorphologyAnalyzer


class OsteoporosisInferencer:
    """
    Inference pipeline for Osteoporosis screening using the End-to-End Multi-task Architecture
    combined with Morphological Fusion (MCW / Porosity / Solidity) and XAI Attention/Saliency Maps.
    """

    def __init__(self, weight_path, device=None):
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # 1. Load Multi-task Model
        if str(weight_path).lower().endswith(".onnx"):
            raise ValueError(
                f"OsteoporosisInferencer requires a PyTorch state_dict (.pt/.pth), got ONNX file: {weight_path}"
            )
        if not weight_path or not os.path.exists(weight_path):
            raise FileNotFoundError(f"Multi-task weights not found at {weight_path}")

        self.model = OsteoMAENet(in_channels=3, num_classes=3)
        self.model.load_state_dict(
            torch.load(weight_path, map_location=self.device, weights_only=True)
        )

        self.model.to(self.device)
        self.model.eval()

        # 2. Morphology Analyzer (Used for Morphological Fusion & Explainability)
        self.analyzer = MorphologyAnalyzer(pixels_per_mm=10.0)

        self.transform = T.Compose([T.Resize((256, 512)), T.ToTensor()])

        self.classes = ["Normal", "Osteopenia (경도)", "Osteoporosis (중증)"]

    def _extract_cortical_mask(self, image_pil: Image.Image) -> np.ndarray:
        """
        Extracts a binary cortical bone mask from the inferior mandibular region
        using CLAHE contrast enhancement and adaptive Otsu/morphological thresholding.
        """
        gray = np.array(image_pil.convert("L").resize((512, 256)), dtype=np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

        # Focus on the lower half (inferior mandibular cortex band)
        roi_mask = np.zeros_like(blurred, dtype=np.uint8)
        roi_mask[128:, :] = 255

        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cortical_mask = cv2.bitwise_and(thresh, roi_mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cortical_mask = cv2.morphologyEx(cortical_mask, cv2.MORPH_CLOSE, kernel)
        return cortical_mask

    def _compute_xai_heatmap(
        self, img_t: torch.Tensor, base_rgb: np.ndarray
    ) -> np.ndarray:
        """
        Computes an Explainable AI (XAI) spatial attention + input-gradient saliency map
        overlay (H, W, 3) uint8 highlighting mandibular cortical regions.
        """
        x_in = img_t.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            sa_mod = self.model.patch_embed.spatial_attention
            avg_out = torch.mean(x_in, dim=1, keepdim=True)
            max_out, _ = torch.max(x_in, dim=1, keepdim=True)
            attn_logits = sa_mod.conv1(torch.cat([avg_out, max_out], dim=1))
            spatial_attn = torch.sigmoid(attn_logits).squeeze().detach().cpu().numpy()

            _, class_logits, _ = self.model(x_in, mask=False)
            target_idx = int(torch.argmax(class_logits, dim=1).item())
            score = class_logits[0, target_idx]
            score.backward()

            grad = x_in.grad.detach().abs().max(dim=1)[0].squeeze().cpu().numpy()

        grad_norm = (grad - grad.min()) / (grad.max() - grad.min() + 1e-8)
        attn_norm = (spatial_attn - spatial_attn.min()) / (
            spatial_attn.max() - spatial_attn.min() + 1e-8
        )
        combined = 0.6 * grad_norm + 0.4 * attn_norm
        combined_u8 = np.clip(combined * 255.0, 0, 255).astype(np.uint8)
        heatmap_bgr = cv2.applyColorMap(combined_u8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(base_rgb, 0.60, heatmap_rgb, 0.40, 0)
        return overlay

    def _fuse_morphology_and_logits(
        self, raw_probs: np.ndarray, geom_feats: dict
    ) -> np.ndarray:
        """
        Morphological Fusion: Combines ViT classification probabilities with mandibular
        cortical width (mean_thickness_mm), porosity_index, and solidity to mitigate
        C1 (Normal) false positives and C2 mode collapse.
        """
        mean_t = float(geom_feats.get("mean_thickness_mm", 0.0))
        porosity = float(geom_feats.get("porosity_index", 0.0))
        solidity = float(geom_feats.get("solidity", 0.0))

        if mean_t <= 0.0:
            morph_prior = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float64)
        elif mean_t >= 4.0 and porosity < 0.08 and solidity >= 0.75:
            morph_prior = np.array([0.65, 0.25, 0.10], dtype=np.float64)
        elif mean_t < 2.8 or porosity >= 0.20 or solidity < 0.55:
            morph_prior = np.array([0.10, 0.30, 0.60], dtype=np.float64)
        else:
            morph_prior = np.array([0.25, 0.50, 0.25], dtype=np.float64)

        fused = 0.75 * raw_probs.astype(np.float64) + 0.25 * morph_prior
        fused = fused / (np.sum(fused) + 1e-8)
        return fused

    def predict(self, image_path_or_pil):
        """
        Predicts the osteoporosis risk for a given image using ViT + Morphological Fusion
        and generates an XAI attention/saliency overlay.
        Returns (pred_class, res_dict, xai_overlay_mask, geom_feats).
        """
        if isinstance(image_path_or_pil, str):
            image = Image.open(image_path_or_pil).convert("RGB")
        else:
            image = image_path_or_pil.convert("RGB")

        resized_rgb = np.array(image.resize((512, 256)), dtype=np.uint8)
        img_t = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            _, class_logits, _ = self.model(img_t, mask=False)
            raw_probs = F.softmax(class_logits, dim=1).squeeze().cpu().numpy()

        # 1. Extract cortical mask & compute morphological features (MCW, porosity, solidity)
        cortical_mask = self._extract_cortical_mask(image)
        geom_feats = self.analyzer.analyze_mask(cortical_mask)
        geom_feats["fusion_applied"] = True
        geom_feats["raw_vit_probabilities"] = {
            self.classes[i]: float(raw_probs[i]) for i in range(3)
        }

        # 2. Apply Morphological Fusion
        fused_probs = self._fuse_morphology_and_logits(raw_probs, geom_feats)
        pred_idx = int(np.argmax(fused_probs))

        # 3. Generate XAI Saliency + Spatial Attention Heatmap Overlay
        try:
            mask = self._compute_xai_heatmap(img_t, resized_rgb)
        except Exception:
            mask = cv2.cvtColor(cortical_mask, cv2.COLOR_GRAY2RGB)

        pred_class = self.classes[pred_idx]
        res_dict = {self.classes[i]: float(fused_probs[i]) for i in range(3)}

        return pred_class, res_dict, mask, geom_feats
