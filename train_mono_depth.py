import random
import json
import time
import argparse

import cv2
import torch

import numpy as np
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split


class MonoDepthDataset(Dataset):
    def __init__(self, data_dir: str, target_size=(256, 256)):
        self.root = Path(data_dir)
        self.target_size = target_size
        all_png = sorted(self.root.glob("*.png"))
        self.images = [
            p 
            for p in all_png 
            if "_disparity" not in p.name and "_depth" not in p.name
        ]
        if len(self.images) == 0:
            raise RuntimeError(f"No .png images found in {data_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        stem = img_path.stem

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(
            img, 
            self.target_size, 
            interpolation=cv2.INTER_LINEAR
        )
        img = img.astype(np.float32) / 255.0  # (H,W,3)

        depth_path = self.root / f"{stem}_depth.npy"
        if not depth_path.exists():
            raise RuntimeError(f"Depth file missing for {stem}: {depth_path}")
        depth = np.load(str(depth_path)).astype(np.float32)
        depth = cv2.resize(
            depth,
            self.target_size, 
            interpolation=cv2.INTER_LINEAR
        ) 

        mask = np.isfinite(depth) & (depth > 0)

        # Convert to tensors: 
        # image [C,H,W], depth [1,H,W], mask [1,H,W]
        image_t = torch.from_numpy(img).permute(2, 0, 1).contiguous()  # (3,H,W)
        depth_t = torch.from_numpy(depth).unsqueeze(0).contiguous()    # (1,H,W)
        mask_t = torch.from_numpy(mask.astype(np.uint8)).unsqueeze(0).contiguous() 

        return image_t, depth_t, mask_t, str(img_path)


class SimpleDepthNet(nn.Module):
    def __init__(self):
        super().__init__()
        def conv(in_c, out_c, k=3, p=1, s=1):
            return nn.Sequential(
                nn.Conv2d(
                    in_c, out_c, 
                    kernel_size=k, 
                    stride=s, 
                    padding=p, 
                    bias=True
                ),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        self.enc1 = conv(3, 32)         # 32xHxW
        self.enc2 = conv(32, 64, s=2)   # 64xH/2xW/2
        self.enc3 = conv(64, 128, s=2)  # 128xH/4xW/4
        self.enc4 = conv(128, 256, s=2) # 256xH/8xW/8

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.dec1 = conv(128 + 128, 128)

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.dec2 = conv(64 + 64, 64)

        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.dec3 = conv(32 + 32, 32)

        self.outconv = nn.Conv2d(32, 1, kernel_size=1)  # TODO fix negative values

    def forward(self, x):
        e1 = self.enc1(x)      # (B,32, H,  W)
        e2 = self.enc2(e1)     # (B,64, H/2,W/2)
        e3 = self.enc3(e2)     # (B,128,H/4,W/4)
        e4 = self.enc4(e3)     # (B,256,H/8,W/8)

        d1 = self.up1(e4)      # (B,128,H/4,W/4)
        d1 = torch.cat([d1, e3], dim=1)
        d1 = self.dec1(d1)

        d2 = self.up2(d1)      # (B,64,H/2,W/2)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d3 = self.up3(d2)      # (B,32,H,W)
        d3 = torch.cat([d3, e1], dim=1)
        d3 = self.dec3(d3)

        out = self.outconv(d3)  # (B,1,H,W)
        return out


def masked_log_l1_loss(pred, target, mask, eps=1e-6):
    """
    L1 loss in log: mean(|log(pred) - log(target)|)
    Args:
        pred, target:   tensors (B,1,H,W)
        mask:           byte tensor (B,1,H,W) with 1 for valid
    """
    pred_pos = torch.clamp(pred, min=eps)
    target_pos = torch.clamp(target, min=eps)

    log_pred = torch.log(pred_pos)
    log_target = torch.log(target_pos)

    diff = torch.abs(log_pred - log_target) * mask.float()
    denom = mask.float().sum()
    if denom == 0:
        return torch.tensor(0.0, device=pred.device)
    return diff.sum() / denom


def compute_metrics(pred, target, mask, eps=1e-6):
    """
    Compute common depth metrics over masked pixels.
    
    Args:
        pred, target: cpu numpy arrays (H,W) or (B,1,H,W)
        mask: byte mask
    Returns:
        Dict: AbsRel, RMSE, RMSE_log, delta1, delta2, delta3
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    if pred.ndim == 4:
        pred = pred[:, 0, :, :]
        target = target[:, 0, :, :]
        mask = mask[:, 0, :, :]

        metrics = {
            'AbsRel': 0, 
            'RMSE': 0, 
            'RMSE_log': 0, 
            'delta1': 0, 
            'delta2': 0, 
            'delta3': 0
        }
        count = 0
        for p, t, m in zip(pred, target, mask):
            valid = m.astype(bool)
            if valid.sum() == 0:
                continue
            mp = p[valid]
            mt = t[valid]
            mp = np.maximum(mp, eps)
            mt = np.maximum(mt, eps)

            absrel = np.mean(np.abs(mt - mp) / mt)
            rmse = np.sqrt(np.mean((mt - mp) ** 2))
            rmse_log = np.sqrt(np.mean((np.log(mp) - np.log(mt)) ** 2))
            ratio = np.maximum(mt / mp, mp / mt)
            delta1 = np.mean(ratio < 1.25)
            delta2 = np.mean(ratio < 1.25 ** 2)
            delta3 = np.mean(ratio < 1.25 ** 3)

            metrics['AbsRel']   += absrel
            metrics['RMSE']     += rmse
            metrics['RMSE_log'] += rmse_log
            metrics['delta1']   += delta1
            metrics['delta2']   += delta2
            metrics['delta3']   += delta3
            count += 1

        if count == 0:
            return {k: 0.0 for k in metrics}
        
        for k in metrics:
            metrics[k] /= count

        return metrics

    else:
        valid = mask.astype(bool)
        if valid.sum() == 0:
            return {
                'AbsRel': 0.0, 
                'RMSE': 0.0, 
                'RMSE_log': 0.0, 
                'delta1': 0.0, 
                'delta2': 0.0, 
                'delta3': 0.0
            }
        
        pred = np.maximum(pred, eps)
        target = np.maximum(target, eps)
        mp = pred[valid]
        mt = target[valid]

        absrel = np.mean(np.abs(mt - mp) / mt)
        rmse = np.sqrt(np.mean((mt - mp) ** 2))
        rmse_log = np.sqrt(np.mean((np.log(mp) - np.log(mt)) ** 2))
        ratio = np.maximum(mt / mp, mp / mt)
        delta1 = np.mean(ratio < 1.25)
        delta2 = np.mean(ratio < 1.25 ** 2)
        delta3 = np.mean(ratio < 1.25 ** 3)

        return {
            'AbsRel': absrel, 
            'RMSE': rmse, 
            'RMSE_log': rmse_log, 
            'delta1': delta1, 
            'delta2': delta2, 
            'delta3': delta3
        }


def train_loop(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    full_ds = MonoDepthDataset(
        args.data_dir, 
        target_size=(args.img_size, args.img_size)
    )
    n = len(full_ds)
    n_val = max(1, int(n * args.val_split))
    n_train = n - n_val
    train_ds, val_ds = random_split(
        full_ds,
        [n_train, n_val], 
        generator=torch.Generator().manual_seed(42)
    )
    print(f"Found {n} samples -> train {n_train}, val {n_val}")

    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch, 
        shuffle=True, 
        num_workers=0, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch, 
        shuffle=False, 
        num_workers=0, 
        pin_memory=True
    )

    model = SimpleDepthNet().to(device)
    optimizer = optim.Adam(
        model.parameters(), 
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    best_val_score = float('inf')
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        iters = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]")
        for images, depths, masks, _paths in pbar:
            images = images.to(device)
            depths = depths.to(device)
            masks = masks.to(device)

            preds_raw = model(images)  # (B,1,H,W)
            if preds_raw.shape != depths.shape:
                preds = torch.nn.functional.interpolate(
                    preds_raw, 
                    size=depths.shape[2:], 
                    mode='bilinear', 
                    align_corners=False
                )
            else:
                preds = preds_raw

            preds_pos = torch.clamp(preds, min=1e-6)

            loss = masked_log_l1_loss(preds_pos, depths, masks, eps=1e-6)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            iters += 1
            pbar.set_postfix({'loss': f"{running_loss / iters:.5f}"})

        epoch_train_loss = running_loss / max(1, iters)
        history['train_loss'].append(epoch_train_loss)

        model.eval()
        val_loss = 0.0
        val_iters = 0
        all_metrics = {
            'AbsRel': 0.0, 
            'RMSE': 0.0, 
            'RMSE_log': 0.0, 
            'delta1': 0.0, 
            'delta2': 0.0, 
            'delta3': 0.0
        }

        with torch.no_grad():
            for images, depths, masks, _paths in tqdm(
                                    val_loader, 
                                    desc=f"Epoch {epoch}/{args.epochs} [val]"
                                                    ):
                images = images.to(device)
                depths = depths.to(device)
                masks = masks.to(device)

                preds_raw = model(images)
                if preds_raw.shape != depths.shape:
                    preds = torch.nn.functional.interpolate(
                        preds_raw, 
                        size=depths.shape[2:], 
                        mode='bilinear', 
                        align_corners=False
                    )
                else:
                    preds = preds_raw
                preds_pos = torch.clamp(preds, min=1e-6)

                loss = masked_log_l1_loss(preds_pos, depths, masks, eps=1e-6)
                val_loss += loss.item()
                val_iters += 1

                metrics = compute_metrics(
                    preds_pos.cpu(), 
                    depths.cpu(), 
                    masks.cpu()
                )
                for k in all_metrics:
                    all_metrics[k] += metrics[k]

        if val_iters > 0:
            epoch_val_loss = val_loss / val_iters
            history['val_loss'].append(epoch_val_loss)
            for k in all_metrics:
                all_metrics[k] /= max(1, val_iters)
        else:
            epoch_val_loss = 0.0

        print(f"Epoch {epoch} summary: train_loss={epoch_train_loss:.5f}, \
                                       val_loss={epoch_val_loss:.5f}")
        print(" Val metrics:", {
                    k: f"{all_metrics[k]:.4f \
                    }" for k in all_metrics
                }
            )

        ckpt_dir = Path(args.ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"epoch_{epoch:03d}.pth"
        torch.save(
            {
                'epoch': epoch, 
                'model_state': model.state_dict(), 
                'optimizer_state': optimizer.state_dict()
            }, str(ckpt_path)
        )

        if epoch_val_loss < best_val_score:
            best_val_score = epoch_val_loss
            best_path = ckpt_dir / "best_model.pth"
            torch.save(
                {
                    'epoch': epoch, 
                    'model_state': model.state_dict(), 
                    'optimizer_state': optimizer.state_dict()
                }, str(best_path)
            )
            print(f" -> New best model saved (val_loss {best_val_score:.5f})")

    hist_path = ckpt_dir / "train_history.json"
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)

    print("Training complete. Best val loss:", best_val_score)
    print("Checkpoints and history saved to", ckpt_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
                        "--data_dir", 
                        type=str, 
                        default="distance_dataset", 
                        help="folder with images & *_depth.npy"
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument(
                        "--img_size", 
                        type=int, 
                        default=256, 
                        help="Image size"
    )
    parser.add_argument(
                        "--val_split", 
                        type=float, 
                        default=0.1, 
                        help="fraction for validation set"
    )
    parser.add_argument(
                        "--ckpt_dir", 
                        type=str, 
                        default="checkpoints", 
                        help="Checkpoint directory"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_loop(args)
