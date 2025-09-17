import cv2
import numpy as np
import json

from typing import Optional, Tuple, Dict, Any


class StereoCalibrator:
    def __init__(
            self, 
            chessboard_size: Tuple[int, int] = (9, 6), 
            square_size: float = 0.025
        ):
        """
        Calibrator init
        
        Args:
            chessboard_size: chessboard size aka number of interior corners
            square_size: size of a chessboard square [m]
        """
        self.chessboard_size = chessboard_size
        self.square_size = square_size
        
        # Criteria for finding angles
        self.criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001
        )
        
        # Preparing object points
        self.objp = np.zeros(
            (
                chessboard_size[0] * chessboard_size[1], 3
            ), np.float32
        )
        self.objp[:, :2] = np.mgrid[
            0:chessboard_size[0], 
            0:chessboard_size[1]
        ].T.reshape(-1, 2)
        self.objp *= square_size
        
        # Arrays for storing points
        self.obj_points = []  # 3D dots in the real world
        self.img_points_left = []  # 2D dots on the left image
        self.img_points_right = []  # 2D dots on the right image
        
    def find_chessboard_corners(self, image):
        """
        Finding the corners of a chessboard in an image
        
        Args:
            image: input image
            
        Returns:
            Found angles or None
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(
            gray, 
            self.chessboard_size, 
            None
        )
        
        if ret:
            corners_refined = cv2.cornerSubPix(
                gray, corners, 
                (11, 11), (-1, -1), 
                self.criteria
            )
            return corners_refined
        return None
    
    def process_images(self, left_images, right_images):
        """
        Processing image pairs for calibration
        
        Args:
            left_images: list of paths to images from the left camera
            right_images: list of paths to images from the right camera
        """
        for left_path, right_path in zip(left_images, right_images):
            left_img = cv2.imread(left_path)
            right_img = cv2.imread(right_path)
            
            if left_img is None or right_img is None:
                print(f"Failed to load images: {left_path} or {right_path}")
                continue
            
            # Finding corners in both images
            corners_left = self.find_chessboard_corners(left_img)
            corners_right = self.find_chessboard_corners(right_img)
            
            if corners_left is not None and corners_right is not None:
                self.obj_points.append(self.objp)
                self.img_points_left.append(corners_left)
                self.img_points_right.append(corners_right)
                
                cv2.drawChessboardCorners(
                    left_img, 
                    self.chessboard_size, 
                    corners_left, 
                    True
                )
                cv2.drawChessboardCorners(
                    right_img, 
                    self.chessboard_size, 
                    corners_right, 
                    True
                )
                
                mosaic = np.hstack((left_img, right_img))
                cv2.imshow('Chessboard Corners', mosaic)
                cv2.waitKey(500)
        
        cv2.destroyAllWindows()
        print(f"Processed {len(self.obj_points)} pairs of images")
    
    def calibrate(self, image_size):
        """
        Stereo camera calibration

        Args:
            image_size: image size (width, height)

        Returns:
            calibration_data: dictionary with calibration parameters
        """
        if len(self.obj_points) < 5:
            print("Not enough images for calibration")
            return None
        
        print("Calibrating the left camera...")
        ret_left, mtx_left, dist_left, \
        rvecs_left, tvecs_left = cv2.calibrateCamera(
            self.obj_points, 
            self.img_points_left, 
            image_size, 
            None, 
            None
        )
        
        print("Calibrating the right camera...")
        ret_right, mtx_right, dist_right,\
        rvecs_right, tvecs_right = cv2.calibrateCamera(
            self.obj_points, 
            self.img_points_right, 
            image_size, 
            None, 
            None
        )
        
        print("Stereo calibration...")
        flags = cv2.CALIB_FIX_INTRINSIC
        ret_stereo, mtx_left, dist_left,\
        mtx_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
            self.obj_points, 
            self.img_points_left, 
            self.img_points_right,
            mtx_left, dist_left, 
            mtx_right, dist_right, 
            image_size,
            criteria=(
                cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5
            ),
            flags=flags
        )
        
        print("Stereorectification...")
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            mtx_left, dist_left, mtx_right, dist_right,
            image_size, R, T, alpha=0)
        
        # Calculation of transformation maps for rectification
        left_map_x, left_map_y = cv2.initUndistortRectifyMap(
            mtx_left, dist_left, R1, P1, image_size, cv2.CV_32FC1)
        
        right_map_x, right_map_y = cv2.initUndistortRectifyMap(
            mtx_right, dist_right, R2, P2, image_size, cv2.CV_32FC1)
        
        # Saving parameters
        calibration_data = {
            'camera_matrix_left': mtx_left,
            'dist_coeffs_left': dist_left,
            'camera_matrix_right': mtx_right,
            'dist_coeffs_right': dist_right,
            'rotation_matrix': R,
            'translation_vector': T,
            'essential_matrix': E,
            'fundamental_matrix': F,
            'left_map_x': left_map_x,
            'left_map_y': left_map_y,
            'right_map_x': right_map_x,
            'right_map_y': right_map_y,
            'roi_left': roi1,
            'roi_right': roi2,
            'q_matrix': Q,
            'baseline': np.linalg.norm(T),  # distance between cameras
            'focal_length': mtx_left[0, 0]  # focal length
        }
        
        return calibration_data
    
    def save_calibration(self, calibration_data, filename):
        """
        Saving calibration parameters to a file

        Args:
            calibration_data: Dictionary with calibration parameters
            filename: File name to save
        """
        data_to_save = {}
        for key, value in calibration_data.items():
            if isinstance(value, np.ndarray):
                data_to_save[key] = value.tolist()
            else:
                data_to_save[key] = value
        
        with open(filename, 'w') as f:
            json.dump(data_to_save, f, indent=2)
        
        print(f"Calibration parameters are saved in {filename}")