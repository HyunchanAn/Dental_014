import cv2
import numpy as np
from skimage.morphology import skeletonize, medial_axis


class MorphologyAnalyzer:
    def __init__(self, pixels_per_mm=10.0):
        self.pixels_per_mm = pixels_per_mm

    def analyze_mask(self, mask: np.ndarray):
        """
        Analyze the binary mask (0 or 255) of the cortical bone.
        Returns a dictionary of morphological features.
        """
        if mask.max() > 1:
            mask = (mask > 127).astype(np.uint8)

        # Ensure mask is not empty
        if np.sum(mask) == 0:
            return {
                "mean_thickness_mm": 0.0,
                "std_thickness_mm": 0.0,
                "min_thickness_mm": 0.0,
                "porosity_index": 0.0,
                "solidity": 0.0,
            }

        # 1. Calculate Medial Axis and Distance Transform
        skel, distance = medial_axis(mask, return_distance=True)

        # The distance to the closest boundary is the radius. Thickness = 2 * radius.
        thickness_pixels = distance[skel] * 2.0
        thickness_mm = thickness_pixels / self.pixels_per_mm

        if len(thickness_mm) == 0:
            mean_thickness = 0.0
            std_thickness = 0.0
            min_thickness = 0.0
        else:
            # Ignore the extreme 5% to avoid edge artifacts
            p5 = np.percentile(thickness_mm, 5)
            p95 = np.percentile(thickness_mm, 95)
            valid_thickness = thickness_mm[(thickness_mm >= p5) & (thickness_mm <= p95)]

            if len(valid_thickness) == 0:
                valid_thickness = thickness_mm

            mean_thickness = np.mean(valid_thickness)
            std_thickness = np.std(valid_thickness)
            min_thickness = np.min(valid_thickness)

        # 2. Calculate Porosity and Solidity
        # Find contours
        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )

        holes_area = 0
        outer_area = 0
        if hierarchy is not None:
            for i in range(len(contours)):
                area = cv2.contourArea(contours[i])
                if hierarchy[0][i][3] != -1:
                    # It's a hole (child contour)
                    holes_area += area
                else:
                    outer_area += area

        # Porosity index: area of holes / total area of bone
        porosity_index = holes_area / (outer_area + 1e-5)

        # Solidity: Bone Area / Convex Hull Area
        # Find the largest contour to compute convex hull
        if len(contours) > 0:
            # Filter outer contours
            outer_contours = [
                contours[i] for i in range(len(contours)) if hierarchy[0][i][3] == -1
            ]
            if outer_contours:
                largest_contour = max(outer_contours, key=cv2.contourArea)
                hull = cv2.convexHull(largest_contour)
                hull_area = cv2.contourArea(hull)
                bone_area = cv2.contourArea(largest_contour) - holes_area
                solidity = bone_area / (hull_area + 1e-5)
            else:
                solidity = 0.0
        else:
            solidity = 0.0

        return {
            "mean_thickness_mm": float(mean_thickness),
            "std_thickness_mm": float(std_thickness),
            "min_thickness_mm": float(min_thickness),
            "porosity_index": float(porosity_index),
            "solidity": float(solidity),
        }
