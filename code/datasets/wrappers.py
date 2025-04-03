import functools
import random
import math
from PIL import Image
import pdb
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from io import BytesIO
import pdb
from datasets import register
from utils import to_pixel_samples


@register('sr-implicit-paired')
class SRImplicitPaired(Dataset):

    def __init__(self, dataset, inp_size=None, augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]

        s = img_hr.shape[-2] // img_lr.shape[-2] # assume int scale
        if self.inp_size is None:
            h_lr, w_lr = img_lr.shape[-2:]
            img_hr = img_hr[:, :h_lr * s, :w_lr * s]
            crop_lr, crop_hr = img_lr, img_hr
        else:
            w_lr = self.inp_size
            x0 = random.randint(0, img_lr.shape[-2] - w_lr)
            y0 = random.randint(0, img_lr.shape[-1] - w_lr)
            crop_lr = img_lr[:, x0: x0 + w_lr, y0: y0 + w_lr]
            w_hr = w_lr * s
            x1 = x0 * s
            y1 = y0 * s
            crop_hr = img_hr[:, x1: x1 + w_hr, y1: y1 + w_hr]

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        scale = torch.ones_like(hr_coord)
        scale[:, 0] *= 1 / crop_hr.shape[-2]
        scale[:, 1] *= 1 / crop_hr.shape[-1]

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'scale': scale,
            'gt': hr_rgb
        }


def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, Image.BICUBIC)(
            transforms.ToPILImage()(img)))

def JPEGcompression(image, qf=10):
    # qf = random.randrange(10, 75)
    outputIoStream = BytesIO()
    image.save(outputIoStream, "JPEG", quality=qf, optimice=True)
    outputIoStream.seek(0)
    # pdb.set_trace()
    return Image.open(outputIoStream)

def resize_compress_fn(img, size):
    pdb.set_trace()
    return transforms.ToTensor()(
        JPEGcompression(transforms.Resize(size, Image.BICUBIC)(
            transforms.ToPILImage()(img)))
        )
        


@register('sr-implicit-downsampled')
class SRImplicitDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None, compression=False):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q
        self.compression = compression

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        s = random.uniform(self.scale_min, self.scale_max)

        if self.inp_size is None:
            h_lr = math.floor(img.shape[-2] / s + 1e-9)
            w_lr = math.floor(img.shape[-1] / s + 1e-9)
            img = img[:, :round(h_lr * s), :round(w_lr * s)] # assume round int
            if not self.compression:
                img_down = resize_fn(img, (h_lr, w_lr))
            else:
                img_down = resize_compress_fn(img, (h_lr, w_lr))
                pdb.set_trace()
            crop_lr, crop_hr = img_down, img
        else:
            w_lr = self.inp_size
            w_hr = round(w_lr * s)
            x0 = random.randint(0, img.shape[-2] - w_hr)
            y0 = random.randint(0, img.shape[-1] - w_hr)
            crop_hr = img[:, x0: x0 + w_hr, y0: y0 + w_hr]
            # pdb.set_trace()
            if not self.compression:
                crop_lr = resize_fn(crop_hr, w_lr)
            else:
                crop_lr = resize_compress_fn(crop_hr, w_lr)

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        scale = torch.ones_like(hr_coord)
        # scale[:, 0] *= 2 / crop_hr.shape[-2]
        # scale[:, 1] *= 2 / crop_hr.shape[-1]
        scale[:, 0] *= 1 / crop_hr.shape[-2]
        scale[:, 1] *= 1 / crop_hr.shape[-1]
        # pdb.set_trace()
        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'scale': scale,
            'gt': hr_rgb
        }


@register('sr-implicit-uniform-varied')
class SRImplicitUniformVaried(Dataset):

    def __init__(self, dataset, size_min, size_max=None,
                 augment=False, gt_resize=None, sample_q=None):
        self.dataset = dataset
        self.size_min = size_min
        if size_max is None:
            size_max = size_min
        self.size_max = size_max
        self.augment = augment
        self.gt_resize = gt_resize
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]
        p = idx / (len(self.dataset) - 1)
        w_hr = round(self.size_min + (self.size_max - self.size_min) * p)
        img_hr = resize_fn(img_hr, w_hr)

        if self.augment:
            if random.random() < 0.5:
                img_lr = img_lr.flip(-1)
                img_hr = img_hr.flip(-1)

        if self.gt_resize is not None:
            img_hr = resize_fn(img_hr, self.gt_resize)

        hr_coord, hr_rgb = to_pixel_samples(img_hr)

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        scale = torch.ones_like(hr_coord)
        scale[:, 0] *= 1 / img_hr.shape[-2]
        scale[:, 1] *= 1 / img_hr.shape[-1]

        return {
            'inp': img_lr,
            'coord': hr_coord,
            'scale': scale,
            'gt': hr_rgb
        }

@register('sr-implicit-downsampled-allqp')
class SRImplicitDownsampledAllQP(Dataset):
    def __init__(self, dataset, inp_size, scale=3, augment=False, sample_q=None):
        """
        Args:
            dataset (Dataset): Underlying paired dataset returning a dict with keys:
                               'hr' (HR image tensor) and 'lr' (dict mapping QP to LR image tensor).
            inp_size (int): Crop size for the LR patch.
            scale (int): Upscaling factor. (HR = LR * scale)
            augment (bool): Whether to apply random flips.
            sample_q (int, optional): If provided, randomly sample this many pixels from the HR crop.
        """
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale = scale
        self.augment = augment
        self.sample_q = sample_q
        # Retrieve QP keys from the first sample.
        sample0 = self.dataset[0]
        self.qp_keys = list(sample0['lr'].keys())
        self.num_qp = len(self.qp_keys)

    def __len__(self):
        return len(self.dataset) * self.num_qp

    def __getitem__(self, idx):
        sample_idx = idx // self.num_qp
        qp_idx = idx % self.num_qp
        sample = self.dataset[sample_idx]
        hr = sample['hr']          # HR tensor: [C, H, W]
        lr_all = sample['lr']
        qp = self.qp_keys[qp_idx]
        lr = lr_all[qp]

        if self.augment:
            if random.random() < 0.5:
                hr = torch.flip(hr, dims=[-1])
                lr = torch.flip(lr, dims=[-1])
            if random.random() < 0.5:
                hr = torch.flip(hr, dims=[-2])
                lr = torch.flip(lr, dims=[-2])

        # Random crop on LR image.
        _, H_lr, W_lr = lr.shape
        if H_lr < self.inp_size or W_lr < self.inp_size:
            raise ValueError("LR image is smaller than the specified inp_size")
        x_lr = random.randint(0, H_lr - self.inp_size)
        y_lr = random.randint(0, W_lr - self.inp_size)
        lr_crop = lr[:, x_lr: x_lr+self.inp_size, y_lr: y_lr+self.inp_size]

        x_hr = x_lr * self.scale
        y_hr = y_lr * self.scale
        inp_size_hr = self.inp_size * self.scale
        hr_crop = hr[:, x_hr: x_hr+inp_size_hr, y_hr: y_hr+inp_size_hr]

        hr_coord, hr_rgb = to_pixel_samples(hr_crop.contiguous())
        scale_tensor = torch.ones_like(hr_coord)
        scale_tensor[:, 0] *= 1.0 / hr_crop.shape[-2]
        scale_tensor[:, 1] *= 1.0 / hr_crop.shape[-1]

        if self.sample_q is not None and self.sample_q < hr_coord.shape[0]:
            perm = torch.randperm(hr_coord.shape[0])[:self.sample_q]
            hr_coord = hr_coord[perm]
            hr_rgb = hr_rgb[perm]
            scale_tensor = scale_tensor[perm]

        return {
            'inp': lr_crop,        # LR input patch: shape [C, inp_size, inp_size]
            'coord': hr_coord,     # HR pixel coordinates (normalized): shape [sample_q, 2]
            'scale': scale_tensor, # Normalization factors: shape [sample_q, 2]
            'gt': hr_rgb           # Ground truth HR pixel values: shape [sample_q, C]
        }