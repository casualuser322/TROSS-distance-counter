import cv2
import numpy as np
import json
import time
from pathlib import Path

from enum import Enum
from typing import Optional, Tuple, Dict, Any

from calibrator import StereoCalibrator
from dataset_collector import DistanceDatasetCollector

class StereoMode(Enum):
    REAL_TIME = "real_time"
    VIDEO_FILE = "video_file"

class StereoVision:
    def __init__(
            self, 
            calibration_file: str = "calibration.json",
            dataset_collector: Optional[DistanceDatasetCollector] = None
        ):
        self.calibration_data = self.load_calibration(calibration_file)
        self.mode = None
        self.left_cap = None
        self.right_cap = None
        self.is_initialized = False
        self.dataset_collector = dataset_collector 
        
        self.set_stereo_params()
        
    def load_calibration(self, calibration_file: str) -> Dict[str, Any]:
        """
        Loading calibration parameters from a file
        
        Args:
            calibration_file: path to the JSON file 
                                with calibration parameters
        """
        try:
            with open(calibration_file, 'r') as f:
                data = json.load(f)
                
            for key in data:
                if isinstance(data[key], list):
                    data[key] = np.array(data[key])
                    
            return data
        except FileNotFoundError:
            print(f"Calibration file {calibration_file} not found")
            return {}
        except json.JSONDecodeError:
            print(f"Error reading calibration file {calibration_file}")
            return {}
    
    def set_stereo_params(self, 
                         min_disparity: int       = 0,
                         num_disparities: int     = 64, 
                         block_size: int          = 15,
                         speckle_window_size: int = 100,
                         speckle_range: int       = 32,
                         disp12_max_diff: int     = 1,
                         pre_filter_cap: int      = 63,
                         uniqueness_ratio: int    = 15,
                         window_size: int         = 3,
                         mode: str                = 'SGBM'
        ):
        """
        Setting parameters for the stereo matching algorithm

        Args:
            min_disparity:       minimum disparity value
            num_disparities:     number of disparity levels (n//16)
            block_size:          block size for matching
            speckle_window_size: maximum speckle size
            speckle_range:       maximum disparity within a speckle
            disp12_max_diff:     maximum disparity difference
            pre_filter_cap:      prefilter
            uniqueness_ratio:    uniqueness threshold
            window_size:         window size for SGBM
            mode:                operating mode ('BM' or 'SGBM')
        """
        self.stereo_params = {
            'minDisparity':      min_disparity,
            'numDisparities':    num_disparities,
            'blockSize':         block_size,
            'speckleWindowSize': speckle_window_size,
            'speckleRange':      speckle_range,
            'disp12MaxDiff':     disp12_max_diff,
            'preFilterCap':      pre_filter_cap,
            'uniquenessRatio':   uniqueness_ratio,
            'P1':                8 * 3 * window_size ** 2,
            'P2':                32 * 3 * window_size ** 2,
            'mode':              mode
        }
        
        if mode == 'SGBM':
            self.stereo = cv2.StereoSGBM_create(
                minDisparity=min_disparity,
                numDisparities=num_disparities,
                blockSize=block_size,
                P1=8 * 3 * window_size ** 2,
                P2=32 * 3 * window_size ** 2,
                disp12MaxDiff=disp12_max_diff,
                uniquenessRatio=uniqueness_ratio,
                speckleWindowSize=speckle_window_size,
                speckleRange=speckle_range,
                preFilterCap=pre_filter_cap
            )
        else:  # BM 
            self.stereo = cv2.StereoBM_create(
                numDisparities=num_disparities,
                blockSize=block_size
            )
    
    def initialize_real_time(
            self, 
            left_cam_id: int = 0, 
            right_cam_id: int = 1
        ):
        """
        Initialization for working with real cameras
        
        Args:
            left_cam_id:  ID left camera
            right_cam_id: ID right camera
        """
        self.mode = StereoMode.REAL_TIME
        
        self.left_cap = cv2.VideoCapture(left_cam_id)
        self.right_cap = cv2.VideoCapture(right_cam_id)
        
        if not self.left_cap.isOpened():
            raise ValueError(f"Failed to open left camera \
                             с ID {left_cam_id}")
        
        if not self.right_cap.isOpened():
            raise ValueError(f"Failed to open right camera \
                             с ID {right_cam_id}")
        
        self.left_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.left_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.right_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.right_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.is_initialized = True
        print(f"Real-time mode initialized:\
                    cameras {left_cam_id} and {right_cam_id}")
    
    def initialize_video_files(
            self, 
            left_video_path: str, 
            right_video_path: str
        ):
        """
        Initialization for working with video filess
        
        Args:
            left_video_path:  path to the video from the left camera
            right_video_path: path to the video from the right camera
        """
        self.mode = StereoMode.VIDEO_FILE
        
        if not Path(left_video_path).exists():
            raise FileNotFoundError(f"Videofile not found: {left_video_path}")
        
        if not Path(right_video_path).exists():
            raise FileNotFoundError(f"Videofile not found:\
                                    {right_video_path}")
        
        self.left_cap = cv2.VideoCapture(left_video_path)
        self.right_cap = cv2.VideoCapture(right_video_path)
        
        if not self.left_cap.isOpened():
            raise ValueError(f"Failed to open videofile:\
                              {left_video_path}")
        
        if not self.right_cap.isOpened():
            raise ValueError(f"Failed to open videofile:\
                              {right_video_path}")
        
        self.is_initialized = True
        print(f"Videofile mode initialized: \
              {left_video_path} and {right_video_path}")
    
    def get_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Receiving frames from stream
        """
        if not self.is_initialized:
            print("The system has not been initialized")
            return None, None
        
        ret_left, frame_left = self.left_cap.read()
        ret_right, frame_right = self.right_cap.read()
        
        if not ret_left or not ret_right:
            if self.mode == StereoMode.VIDEO_FILE:
                print("The end of videofiles")
            return None, None
        
        return frame_left, frame_right
    
    def rectify_frames(
            self, 
            frame_left: np.ndarray, 
            frame_right: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Frame rectification, distortion correction and alignment
        
        Args:
            frame_left:  left camera shot
            frame_right: frame from the right camera
        """
        if not self.calibration_data:
            print("No calibration data available")
            return frame_left, frame_right
        
        left_map_x = self.calibration_data.get('left_map_x')
        left_map_y = self.calibration_data.get('left_map_y')
        right_map_x = self.calibration_data.get('right_map_x')
        right_map_y = self.calibration_data.get('right_map_y')
        
        if left_map_x is None or left_map_y is None \
            or right_map_x is None or right_map_y is None:
            print("No rectification maps found")
            return frame_left, frame_right
        
        left_rectified = cv2.remap(
            frame_left, 
            left_map_x, 
            left_map_y, 
            cv2.INTER_LINEAR
        )
        right_rectified = cv2.remap(
            frame_right, 
            right_map_x, 
            right_map_y, 
            cv2.INTER_LINEAR
        )
        
        return left_rectified, right_rectified
    
    def compute_disparity(
            self, 
            left_frame: np.ndarray, 
            right_frame: np.ndarray
        ) -> np.ndarray:
        """
        Calculate the disparity map

        Args:
            left_frame:  rectified frame from the left camera
            right_frame: rectified frame from the right camera

        Returns:
            Disparity map
        """

        if len(left_frame.shape) == 3:
            left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
        else:
            left_gray = left_frame
            right_gray = right_frame
        
        disparity = self.stereo.compute(left_gray, right_gray)
        
        # Normalization for visualization
        disparity_normalized = cv2.normalize(
            disparity, 
            None, 
            0, 255, 
            cv2.NORM_MINMAX
        )
        disparity_normalized = np.uint8(disparity_normalized)
        
        return disparity_normalized
    
    def calculate_distance(self, disparity: float) -> float:
        """
        Calculating the distance to an object based on disparity

        Args:
            disparity: disparity value in pixels

        Returns:
            Distance to the object in meters
        """
        if disparity <= 0:
            return float('inf')
        
        # distance = (baseline * focal_length) / disparity
        baseline = self.calibration_data.get(
            'baseline', 0.12    # [m] (default: 12sm)
        )  
        focal_length = self.calibration_data.get('focal_length', 1250)
        
        distance = (baseline * focal_length) / disparity
        return distance
    
    def get_distance_at_point(
            self, 
            disparity_map: np.ndarray, 
            x: int, y: int, 
            window_size: int = 5
        ) -> float:
        """
        Getting the distance to a point in an image

        Args:
            disparity_map: disparity map
            x: X coordinate of the point
            y: Y coordinate of the point
            window_size: Window size for averaging

        Returns:
            Distance to the point in meters
        """
        # Define the area around the point for averaging
        h, w = disparity_map.shape
        x_min = max(0, x - window_size // 2)
        x_max = min(w, x + window_size // 2)
        y_min = max(0, y - window_size // 2)
        y_max = min(h, y + window_size // 2)
        
        # Calculate the average disparity in the region
        disparity_region = disparity_map[y_min:y_max, x_min:x_max]
        mean_disparity = np.mean(disparity_region)
        
        return self.calculate_distance(mean_disparity)
    
    def process_frame(
            self, 
            show_results: bool = True,
            collect_data: bool = True
        ) -> Tuple[Optional[np.ndarray], 
             Optional[np.ndarray],
             Optional[np.ndarray]]:
        """
        Processing a single frame from sources

        Args:
            show_results: whether to show processing results
            collect_data: whether to collect data for training dataset

        Returns:
            Tuple (left frame, right frame, disparity map)
        """
        frame_left, frame_right = self.get_frames()
        
        if frame_left is None or frame_right is None:
            return None, None, None
        
        # Rectification of frames
        left_rect, right_rect = self.rectify_frames(frame_left, frame_right)
        
        disparity = self.compute_disparity(left_rect, right_rect)

        if collect_data and self.dataset_collector is not None:
            focal_length = self.calibration_data.get('focal_length', 1250)
            baseline = self.calibration_data.get('baseline', 0.12)
            self.dataset_collector.save_sample(
                left_rect, disparity, focal_length, baseline
            )
        
        if show_results:
            # Creating a mosaic for display
            top_row = np.hstack((frame_left, frame_right))
            bottom_row = np.hstack((left_rect, right_rect))
            
            # Adding a disparity map
            disparity_color = cv2.applyColorMap(disparity, cv2.COLORMAP_JET)
            if bottom_row.shape[1] > disparity_color.shape[1]:
                padding = np.zeros(
                    (
                        disparity_color.shape[0], 
                        bottom_row.shape[1] - disparity_color.shape[1], 
                        3
                    ), 
                    dtype=np.uint8
                )
                disparity_color = np.hstack((disparity_color, padding))
            
            mosaic = np.vstack((top_row, bottom_row, disparity_color))
            
            cv2.imshow(
                'Stereo Vision: Original | Rectified | Disparity', 
                mosaic
            )
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                return None, None, None
        
        return frame_left, frame_right, disparity
    
    def run(self, collect_data: bool = False):
        if not self.is_initialized:
            print("The system is not initialized")
            return

        if collect_data:
            print("Starting processing with data collection. "
                "Press 'q' to exit, 's' to save a sample.")
        else:
            print("Starting processing. Press 'q' to exit.")

        while True:
            result = self.process_frame(show_results=True, collect_data=collect_data)
            if result[0] is None:
                break

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        self.cleanup()
    
    def cleanup(self):
        if self.left_cap:
            self.left_cap.release()

        if self.right_cap:
            self.right_cap.release()

        cv2.destroyAllWindows()
        print("Resources released")


"""
# Examle
    dataset_collector = DistanceDatasetCollector("stereo_depth_dataset")

    # Working with real cameras
    # stereo = StereoVision("calibration.json", dataset_collector)
    # stereo.initialize_real_time(left_cam_id=0, right_cam_id=1)
    # stereo.run(collect_data=True)
    
    # Working with video files
    # stereo = StereoVision("calibration.json", dataset_collector)
    # stereo.initialize_video_files("left_video.mp4", "right_video.mp4")
    # stereo.run(collect_data=True)
    
    # Calibrating cameras
    # calibrator = StereoCalibrator()
    # left_images = [f"left_{i}.jpg" for i in range(1, 21)]
    # right_images = [f"right_{i}.jpg" for i in range(1, 21)]
    # calibrator.process_images(left_images, right_images)
    # calibration_data = calibrator.calibrate((1280, 720))
    # calibrator.save_calibration(calibration_data, "calibration.json")
"""