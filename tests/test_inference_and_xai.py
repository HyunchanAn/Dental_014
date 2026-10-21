import os
import sys
import tempfile
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from dental_014.inference import OsteoporosisInferencer
from dental_014.multitask_model import OsteoMAENet


def test_osteoporosis_inferencer_morphology_fusion_and_xai():
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "dummy_best.pt")
        model = OsteoMAENet(in_channels=3, num_classes=3)
        torch.save(model.state_dict(), ckpt_path)

        inferencer = OsteoporosisInferencer(weight_path=ckpt_path, device="cpu")

        # Create a synthetic image with a bright cortical bone band in the lower half
        arr = np.zeros((256, 512, 3), dtype=np.uint8)
        arr[170:220, 40:470, :] = 230
        img_pil = Image.fromarray(arr)

        pred_class, probs, xai_overlay, geom_feats = inferencer.predict(img_pil)

        assert pred_class in inferencer.classes
        assert abs(sum(probs.values()) - 1.0) < 1e-4
        assert isinstance(xai_overlay, np.ndarray)
        assert xai_overlay.shape == (256, 512, 3)
        assert xai_overlay.dtype == np.uint8

        assert geom_feats.get("fusion_applied") is True
        assert "mean_thickness_mm" in geom_feats
        assert "porosity_index" in geom_feats
        assert "solidity" in geom_feats
        assert "raw_vit_probabilities" in geom_feats
        assert geom_feats["mean_thickness_mm"] > 0.0
