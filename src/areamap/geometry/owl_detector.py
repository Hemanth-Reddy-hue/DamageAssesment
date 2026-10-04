"""OWLv2 detector for architectural openings in video frames."""

import os
from pathlib import Path
import numpy as np
import cv2
import torch
from typing import List, Dict, Any, Optional

from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection

class OWLDetector:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
        self.model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(self.device)
        self.model.eval()
        
    def detect(self, image_path: str, texts: List[str] = ["a door", "a window"]) -> List[Dict[str, Any]]:
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            return []
            
        inputs = self.processor(text=[texts], images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)
        
        detections = []
        if len(results) > 0:
            result = results[0]
            boxes = result["boxes"].cpu().numpy()
            scores = result["scores"].cpu().numpy()
            labels = result["labels"].cpu().numpy()
            
            for box, score, label in zip(boxes, scores, labels):
                if score > 0.15:  # Empirical threshold for OWLv2
                    detections.append({
                        "label": texts[label].replace("a ", ""),
                        "score": float(score),
                        "box": [float(x) for x in box]  # [xmin, ymin, xmax, ymax]
                    })
                    
        return detections

# Singleton instance
_detector = None

def get_owl_detector() -> OWLDetector:
    global _detector
    if _detector is None:
        _detector = OWLDetector()
    return _detector
