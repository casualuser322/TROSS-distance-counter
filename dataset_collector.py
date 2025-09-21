import os
import json
import time

import cv2
import numpy as np

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


class DistanceDatasetCollector:
    def __init__(self, output_dir: str = "distance_dataset"):
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.annotations_dir = self.output_dir / "annotations"
        
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.annotations_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_index = 0
        self.metadata = []
        
    def save_sample(self, 
                image: np.ndarray, disparity_map: np.ndarray,
                focal_length: float, baseline: float
            ) -> None:
        """Collecting dataset"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"sample_{timestamp}_{self.current_index:06d}"
        
        image_path = self.images_dir / f"{filename}.png"
        cv2.imwrite(str(image_path), image)
        
        # Disparity map
        disparity_path = self.images_dir / f"{filename}_disparity.npy"
        np.save(str(disparity_path), disparity_map)
        
        # Depth map
        depth_map = self.calculate_depth_map(
            disparity_map,
            focal_length, 
            baseline
        )
        depth_path = self.images_dir / f"{filename}_depth.npy"
        np.save(str(depth_path), depth_map)

        metadata = {
            'filename': filename,
            'image_path': str(image_path),
            'disparity_path': str(disparity_path),
            'depth_path': str(depth_path),
            'focal_length': focal_length,
            'baseline': baseline,
            'timestamp': timestamp
        }
        
        self.metadata.append(metadata)
        self.current_index += 1
        
        if self.current_index % 100 == 0:
            self.save_metadata()
    
    def calculate_depth_map(
                        self, 
                        disparity_map: np.ndarray, 
                        focal_length: float, baseline: float
            ) -> np.ndarray:
        """
        Calculating a depth map from a disparity map
        
        Returns:
            Depth map in meters
        """
        disparity_map = np.clip(disparity_map, 1e-6, None)
        depth_map = (focal_length * baseline) / disparity_map
        return depth_map
    
    def save_metadata(self) -> None:
        metadata_path = self.annotations_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def __del__(self):
        self.save_metadata()