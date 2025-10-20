# TROSS distance counter

## Overview

This repository contains two independent but related components:

1. A **stereo calibration module** (`StereoCalibrator`) for computing intrinsic and extrinsic camera parameters using a checkerboard calibration target.
2. A **monocular depth prediction model** (`MonoDepthNet`) implemented in PyTorch. The network is trained to predict dense depth maps from single RGB images.

Both parts can be used separately. The stereo calibration is used to generate accurate disparity-to-depth mappings and rectified stereo image pairs, while the neural network attempts to approximate depth from a single view through supervised learning.

---

## Dependencies

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3-red?logo=pytorch)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.10-green?logo=opencv)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.26-lightgrey?logo=numpy)](https://numpy.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-black?logo=nvidia)](https://developer.nvidia.com/cuda-zone)
[![Training](https://img.shields.io/badge/Deep_Learning-Exhausting-orange?logo=ai)](#)
[![License](https://img.shields.io/badge/License-MIT-grey)](./LICENSE)


Install dependencies:

Everything should be already installed if you use full build

Use this for local tests
```bash
python -m venv .venv
. .venv/bin/activate
pip install torch torchvision opencv-python numpy tqdm
```

---

## Stereo calibration

Stereo calibration estimates the intrinsic matrices $K_l, K_r$, distortion coefficients $D_l, D_r$, and the extrinsic parameters $\text{rotation } R, \text{ translation } T$ between two cameras.

Given sets of corresponding points in the left and right images of a known calibration pattern, it computes these parameters via nonlinear optimization, minimizing the reprojection error of the 3D points onto the image planes.


Each camera is modeled by the pinhole projection:

$$ \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = K \begin{bmatrix} R & T \end{bmatrix} \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix} $$

where  
- $K$ is the intrinsic matrix  
- $R$, $T$ are the rotation and translation from world to camera coordinates  
- $(X, Y, Z)$ are 3D points in world space  
- $(u, v)$ are pixel coordinates

Stereo calibration produces a rectified coordinate system such that corresponding epipolar lines are aligned horizontally. Depth can then be recovered from disparity $d = x_l - x_r$ via:

$$
Z = \frac{f \cdot B}{d}
$$

where $f$ is the focal length and $B$ is the baseline distance between the two cameras.

### Usage

```python
from calibrator import StereoCalibrator
import glob

left_images = sorted(glob.glob("data/left/*.png"))
right_images = sorted(glob.glob("data/right/*.png"))

calib = StereoCalibrator(chessboard_size=(9, 6), square_size=0.025)
calib.process_images(left_images, right_images)

image_size = (1280, 720)
calibration_data = calib.calibrate(image_size)
calib.save_calibration(calibration_data, "stereo_calibration.json")
```


The calibration JSON includes:

- `camera_matrix_left`, `camera_matrix_right` — 3×3 intrinsics  
- `dist_coeffs_left`, `dist_coeffs_right` — distortion coefficients  
- `rotation_matrix`, `translation_vector` — relative transform  
- `essential_matrix`, `fundamental_matrix`  
- `q_matrix` — reprojection matrix  
- `baseline`, `focal_length`  
- `rectification_maps` — pixel remapping for undistortion and rectification  

---

## Monocular depth prediction 

### Architecture

`MonoDepthNet` is a symmetric encoder-decoder convolutional network inspired by U-Net.  
It uses standard convolutional downsampling and transposed convolutional upsampling with skip connections.

```
Input RGB (3×HxW)
↓
[Conv → ReLU → BN] × 4 (downsample)
↓
[UpConv → Cat → Conv] × 3 (upsample)
↓
1×1 Conv → Depth map (1×HxW)
```

All activations are ReLU; batch normalization is applied after each convolution.  
The final layer produces a single-channel depth prediction (not explicitly constrained positive).

### Loss Function

Training minimizes a masked logarithmic L1 loss:

$$
L = \frac{1}{N} \sum_{i \in M} \left| \log(\hat{D}_i + \epsilon) - \log(D_i + \epsilon) \right|
$$

where $M$ is the valid pixel mask and $\epsilon = 10^{-6}$.  
This penalizes multiplicative depth errors rather than additive, emphasizing relative accuracy.

### Metrics

After each validation epoch, the following standard metrics are computed:


| Metric | Formula |
|--------|---------|
| **AbsRel** | $ \frac{1}{N} \sum_i \frac{\|D_i - \hat{D}_i\|}{D_i} $ |
| **RMSE** | $ \sqrt{\frac{1}{N} \sum_i (D_i - \hat{D}_i)^2} $ |
| **RMSE_log** | $ \sqrt{\frac{1}{N} \sum_i (\log D_i - \log \hat{D}_i)^2} $ |
| **δ₁, δ₂, δ₃** | Fraction of pixels s.t. $ \max(\frac{D_i}{\hat{D}_i}, \frac{\hat{D}_i}{D_i}) < 1.25^t $, for $ t = 1, 2, 3 $ |



Each training sample consists of an RGB image and a depth map in NumPy format:

```
distance_dataset/
├── frame_0001.png
├── frame_0001_depth.npy
├── frame_0002.png
├── frame_0002_depth.npy
└── ...
```

The `.npy` file must contain a floating-point depth array in meters.

---

## Training

To train:

```bash
python train_monodepth.py \
    --data_dir distance_dataset/images \
    --epochs 20 \
    --batch 8 \
    --img_size 256 \
    --ckpt_dir checkpoints
```

### Arguments

| Argument | Description | Default |
|-----------|-------------|----------|
| `--data_dir` | Dataset directory | `distance_dataset` |
| `--epochs` | Number of epochs | `20` |
| `--batch` | Batch size | `4` |
| `--lr` | Learning rate | `1e-4` |
| `--weight_decay` | L2 regularization | `1e-6` |
| `--img_size` | Image resize dimension | `256` |
| `--val_split` | Validation set fraction | `0.1` |
| `--ckpt_dir` | Checkpoint directory | `checkpoints` |

### Output

- Per-epoch checkpoints: `epoch_XXX.pth`  
- Best model: `best_model.pth`  
- Training log: `train_history.json`

Each checkpoint includes:
- model weights
- optimizer state
- current epoch number

---

## Computational Notes

- Training time scales approximately linearly with dataset size and resolution.  
- GPU acceleration (CUDA) is automatically used if available.  
- Depth predictions are not guaranteed to be metrically accurate unless trained on physically calibrated depth data.  
- The stereo baseline $B$ and focal length $f$ from calibration can be used to convert disparity maps to metric depth for ground truth supervision.


## TODO
 - Normalize depths before training to improve stability.
 - Add augmentations (torchvision.transforms or Albumentations).
 - Add gradient clipping to avoid gradient explosions.