import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A

IMG_SIZE = 256
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _to_tensor(img: np.ndarray, mask: np.ndarray):
    """Convert HxWxC uint8 image and HxW float32 mask to tensors
    without using torch.from_numpy (avoids numpy-version bridge issues)."""
    # img: HxWx3 float32 after Normalize
    img_t  = torch.tensor(img.transpose(2, 0, 1).copy())   # CxHxW
    mask_t = torch.tensor(mask.copy())                      # HxW
    return img_t, mask_t


def get_transforms(mode: str, augment_level: str = 'none') -> A.Compose:
    """
    mode: 'train' or 'val' / 'test'
    augment_level: 'none' | 'light' | 'moderate' | 'heavy' | 'very_heavy'
    """
    post = [
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist()),
    ]

    if mode != 'train' or augment_level == 'none':
        return A.Compose(post)

    aug = []

    # All levels: horizontal and vertical flips
    aug += [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
    ]

    if augment_level in ('moderate', 'heavy', 'very_heavy'):
        aug += [
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        ]

    if augment_level in ('heavy', 'very_heavy'):
        aug += [
            A.ElasticTransform(alpha=120, sigma=6, p=0.3),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
            A.CLAHE(clip_limit=2.0, p=0.3),
        ]

    if augment_level == 'very_heavy':
        aug += [
            A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
            A.Blur(blur_limit=3, p=0.2),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=20, p=0.4),
            A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(8, 32),
                            hole_width_range=(8, 32), p=0.2),
        ]

    return A.Compose(aug + post)


class NerveDataset(Dataset):
    """Ultrasound nerve segmentation dataset."""

    def __init__(self, img_dir: str, mask_dir: str, transform=None):
        self.img_dir   = img_dir
        self.mask_dir  = mask_dir
        self.transform = transform
        self.images    = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name      = self.images[idx]
        img_path  = os.path.join(self.img_dir,  name)
        mask_path = os.path.join(self.mask_dir, name)

        # Load grayscale image → replicate to 3 channels for ImageNet norms
        img  = np.array(Image.open(img_path).convert('RGB'))   # H×W×3 uint8
        mask = np.array(Image.open(mask_path).convert('L'))    # H×W   uint8
        mask = (mask > 127).astype(np.float32)                 # binary {0,1}

        if self.transform:
            out  = self.transform(image=img, mask=mask)
            img, mask = _to_tensor(out['image'], out['mask'])

        return img, mask.unsqueeze(0)   # (C,H,W), (1,H,W)

    def get_filename(self, idx: int) -> str:
        return self.images[idx]


class TestNerveDataset(Dataset):
    """Test dataset — returns image tensor + filename (no mask required)."""

    def __init__(self, img_dir: str, mask_dir: str, transform=None):
        self.img_dir   = img_dir
        self.mask_dir  = mask_dir
        self.transform = transform
        self.images    = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name      = self.images[idx]
        img_path  = os.path.join(self.img_dir,  name)
        mask_path = os.path.join(self.mask_dir, name)

        img  = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'))
        mask = (mask > 127).astype(np.float32)

        if self.transform:
            out  = self.transform(image=img, mask=mask)
            img, mask = _to_tensor(out['image'], out['mask'])

        return img, mask.unsqueeze(0), name
