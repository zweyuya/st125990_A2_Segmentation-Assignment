"""
run.py — Assignment A2-02: Image Segmentation with U-Net
=========================================================
Usage:
  # Train baseline (with skip connections)
  python3 run.py --model unet_resnet18         --dataset oxford_pet --epochs 20 --train

  # Train ablation (no skip connections)
  python3 run.py --model unet_resnet18_no_skip --dataset oxford_pet --epochs 20 --train

  # Evaluate a saved model
  python3 run.py --model unet_resnet18 --weights unet_resnet18_pet.pt --dataset oxford_pet --evaluate
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import OxfordIIITPet
from tqdm import tqdm


# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── Building blocks ───────────────────────────────────────────────────────────
class DoubleConv(nn.Module):
    """Two consecutive Conv2d -> BN -> ReLU blocks."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)


# ── Model 1: U-Net with ResNet-18 encoder + skip connections ──────────────────
class UNetResNet18(nn.Module):
    """U-Net with pretrained ResNet-18 encoder and full skip connections."""

    def __init__(self, n_classes=3, pretrained=True):
        super().__init__()
        weights = 'IMAGENET1K_V1' if pretrained else None
        resnet  = models.resnet18(weights=weights)

        self.stem_conv = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.stem_pool = resnet.maxpool
        self.enc1 = resnet.layer1
        self.enc2 = resnet.layer2
        self.enc3 = resnet.layer3
        self.enc4 = resnet.layer4

        self.bottleneck = DoubleConv(512, 1024)

        self.up4  = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(512 + 512, 512)

        self.up3  = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(256 + 256, 256)

        self.up2  = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(128 + 128, 128)

        self.up1  = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(64 + 64, 64)

        self.up0  = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec0 = DoubleConv(32 + 64, 32)

        self.output = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        s0 = self.stem_conv(x)
        sp = self.stem_pool(s0)
        s1 = self.enc1(sp)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)

        x = self.bottleneck(s4)

        x = self.up4(x);  x = self._cat(x, s4);  x = self.dec4(x)
        x = self.up3(x);  x = self._cat(x, s3);  x = self.dec3(x)
        x = self.up2(x);  x = self._cat(x, s2);  x = self.dec2(x)
        x = self.up1(x);  x = self._cat(x, s1);  x = self.dec1(x)
        x = self.up0(x);  x = self._cat(x, s0);  x = self.dec0(x)

        return self.output(x)

    def _cat(self, x, skip):
        if x.shape[2:] != skip.shape[2:]:
            skip = F.interpolate(skip, size=x.shape[2:])
        return torch.cat([skip, x], dim=1)


# ── Model 2: U-Net with ResNet-18 encoder, NO skip connections (ablation) ─────
class UNetResNet18NoSkip(nn.Module):
    """U-Net with pretrained ResNet-18 encoder but NO skip connections."""

    def __init__(self, n_classes=3, pretrained=True):
        super().__init__()
        weights = 'IMAGENET1K_V1' if pretrained else None
        resnet  = models.resnet18(weights=weights)

        self.stem_conv = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.stem_pool = resnet.maxpool
        self.enc1 = resnet.layer1
        self.enc2 = resnet.layer2
        self.enc3 = resnet.layer3
        self.enc4 = resnet.layer4

        self.bottleneck = DoubleConv(512, 1024)

        self.up4  = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(512, 512)

        self.up3  = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(256, 256)

        self.up2  = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(128, 128)

        self.up1  = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(64, 64)

        self.up0  = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec0 = DoubleConv(32, 32)

        self.output = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        x = self.stem_conv(x)
        x = self.stem_pool(x)
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)

        x = self.bottleneck(x)

        x = self.up4(x);  x = self.dec4(x)
        x = self.up3(x);  x = self.dec3(x)
        x = self.up2(x);  x = self.dec2(x)
        x = self.up1(x);  x = self.dec1(x)
        x = self.up0(x);  x = self.dec0(x)

        return self.output(x)


# ── Model factory ─────────────────────────────────────────────────────────────
def build_model(flag, n_classes=3):
    if flag == 'unet_resnet18':
        return UNetResNet18(n_classes=n_classes, pretrained=True)
    elif flag == 'unet_resnet18_no_skip':
        return UNetResNet18NoSkip(n_classes=n_classes, pretrained=True)
    else:
        raise ValueError(f"Unknown model: '{flag}'. Choose: unet_resnet18 | unet_resnet18_no_skip")


# ── Dataset ───────────────────────────────────────────────────────────────────
class PetSegDataset(Dataset):
    def __init__(self, base, size=128):
        self.ds = base
        self.img_tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.mask_tf = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.PILToTensor(),
        ])
    def __len__(self): return len(self.ds)
    def __getitem__(self, idx):
        img, mask = self.ds[idx]
        img  = self.img_tf(img)
        mask = (self.mask_tf(mask).squeeze(0).long() - 1).clamp(0, 2)
        return img, mask


def get_dataloaders(data_dir='./data', img_size=128, batch_size=16):
    os.makedirs(data_dir, exist_ok=True)
    train_raw = OxfordIIITPet(data_dir, split='trainval', target_types='segmentation', download=True)
    test_raw  = OxfordIIITPet(data_dir, split='test',     target_types='segmentation', download=True)
    train_loader = DataLoader(PetSegDataset(train_raw, img_size), batch_size=batch_size, shuffle=True,  num_workers=2)
    test_loader  = DataLoader(PetSegDataset(test_raw,  img_size), batch_size=batch_size, shuffle=False, num_workers=2)
    print(f'Train: {len(train_raw)} | Test: {len(test_raw)}')
    return train_loader, test_loader


# ── Metric ────────────────────────────────────────────────────────────────────
def compute_iou(pred, target, n_classes=3):
    pred = pred.argmax(dim=1)
    ious = []
    for cls in range(n_classes):
        inter = ((pred == cls) & (target == cls)).sum().float()
        union = ((pred == cls) | (target == cls)).sum().float()
        if union > 0:
            ious.append((inter / union).item())
    return np.mean(ious) if ious else 0.0


# ── Train ─────────────────────────────────────────────────────────────────────
def train(args):
    train_loader, test_loader = get_dataloaders()

    model = build_model(args.model).to(device)
    print(f'Model:  {args.model}')
    print(f'Device: {device}')
    print(f'Params: {sum(p.numel() for p in model.parameters()):,}')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    save_name = f'{args.model}_pet.pt'
    train_losses, val_ious, epoch_times = [], [], []

    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        ep_loss = []
        for imgs, masks in tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}'):
            imgs, masks = imgs.to(device), masks.to(device)
            loss = criterion(model(imgs), masks)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            ep_loss.append(loss.item())

        model.eval()
        ep_iou = []
        with torch.no_grad():
            for imgs, masks in test_loader:
                ep_iou.append(compute_iou(model(imgs.to(device)), masks.to(device)))

        scheduler.step()
        elapsed = time.time() - t0
        train_losses.append(np.mean(ep_loss))
        val_ious.append(np.mean(ep_iou))
        epoch_times.append(elapsed)

        print(f'Epoch {epoch+1:02d} | Loss: {train_losses[-1]:.4f} | mIoU: {val_ious[-1]:.4f} | Time: {elapsed:.1f}s')

    torch.save(model.state_dict(), save_name)
    print(f'\nSaved → {save_name}')
    print(f'Best Val mIoU : {max(val_ious):.4f}')
    print(f'Avg Time/epoch: {np.mean(epoch_times):.1f}s')


# ── Evaluate ──────────────────────────────────────────────────────────────────
def evaluate(args):
    _, test_loader = get_dataloaders()

    model = build_model(args.model).to(device)
    weights_path = args.weights if args.weights else f'{args.model}_pet.pt'
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    print(f'Loaded weights from: {weights_path}')

    ious = []
    with torch.no_grad():
        for imgs, masks in tqdm(test_loader, desc='Evaluating'):
            ious.append(compute_iou(model(imgs.to(device)), masks.to(device)))

    print(f'\nVal mIoU: {np.mean(ious):.4f}')


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description='A2-02 Image Segmentation')
    parser.add_argument('--model',    required=True,
                        choices=['unet_resnet18', 'unet_resnet18_no_skip'],
                        help='Model architecture')
    parser.add_argument('--dataset',  default='oxford_pet',
                        help='Dataset name (default: oxford_pet)')
    parser.add_argument('--epochs',   type=int, default=20,
                        help='Number of training epochs (default: 20)')
    parser.add_argument('--weights',  type=str, default=None,
                        help='Path to saved weights file (for --evaluate)')
    parser.add_argument('--train',    action='store_true',
                        help='Run training')
    parser.add_argument('--evaluate', action='store_true',
                        help='Run evaluation on test set')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(f'Using device: {device}')

    if args.train:
        train(args)
    elif args.evaluate:
        evaluate(args)
    else:
        print('Please specify --train or --evaluate')
