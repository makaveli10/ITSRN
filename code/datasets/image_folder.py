import os
import json
import glob
import random

from PIL import Image

import pickle
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import register


@register('image-folder')
class ImageFolder(Dataset):

    def __init__(self, root_path, split_file=None, split_key=None, first_k=None,
                 repeat=1, cache='none'):
        self.repeat = repeat
        self.cache = cache

        if split_file is None:
            filenames = sorted(os.listdir(root_path))
        else:
            with open(split_file, 'r') as f:
                filenames = json.load(f)[split_key]
        if first_k is not None:
            filenames = filenames[:first_k]

        self.files = []
        for filename in filenames:
            file = os.path.join(root_path, filename)

            if cache == 'none':
                self.files.append(file)

            elif cache == 'bin':
                bin_root = os.path.join(os.path.dirname(root_path),
                    '_bin_' + os.path.basename(root_path))
                if not os.path.exists(bin_root):
                    os.mkdir(bin_root)
                    print('mkdir', bin_root)
                bin_file = os.path.join(
                    bin_root, filename.split('.')[0] + '.pkl')
                if not os.path.exists(bin_file):
                    with open(bin_file, 'wb') as f:
                        pickle.dump(imageio.imread(file), f)
                    print('dump', bin_file)
                self.files.append(bin_file)

            elif cache == 'in_memory':
                self.files.append(transforms.ToTensor()(
                    Image.open(file).convert('RGB')))

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]

        if self.cache == 'none':
            return transforms.ToTensor()(Image.open(x).convert('RGB'))

        elif self.cache == 'bin':
            with open(x, 'rb') as f:
                x = pickle.load(f)
            x = np.ascontiguousarray(x.transpose(2, 0, 1))
            x = torch.from_numpy(x).float() / 255
            return x

        elif self.cache == 'in_memory':
            return x


@register('paired-image-folders')
class PairedImageFolders(Dataset):

    def __init__(self, root_path_1, root_path_2, **kwargs):
        self.dataset_1 = ImageFolder(root_path_1, **kwargs)
        self.dataset_2 = ImageFolder(root_path_2, **kwargs)

    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        return self.dataset_1[idx], self.dataset_2[idx]


@register('paired-image-folder-icme')
class PairedImageFolder(Dataset):
    def __init__(self, hr_root, lr_root, qp_list=None, repeat=1, cache='none'):
        """
        Args:
            hr_root (str): Path to the HR folder (e.g., train_gt). Each subfolder
                           is assumed to be a video (e.g., "0000_1920x1080_30fps").
            lr_root (str): Path to the LR folder (e.g., train_lr_h265). For each HR
                           sample there are multiple LR subfolders named like:
                           "0000_640x360_30fps_qp17", "0000_640x360_30fps_qp22", etc.
            qp_list (list): List of QP strings. Defaults to:
                            ['qp17', 'qp22', 'qp27', 'qp32', 'qp34', 'qp37']
            repeat (int): Repeat factor for the dataset.
            cache (str): Currently only 'none' is implemented.
        """
        self.hr_root = hr_root
        self.lr_root = lr_root
        self.repeat = repeat
        self.cache = cache
        if qp_list is None:
            self.qp_list = ['qp17', 'qp22', 'qp27', 'qp32', 'qp34', 'qp37']
        else:
            self.qp_list = qp_list

        # Build sample list.
        hr_samples = sorted(os.listdir(self.hr_root))
        self.samples = []
        for sample in hr_samples:
            sample_id = sample.split('_')[0]
            hr_sample_path = os.path.join(self.hr_root, sample)
            hr_frames = sorted(glob.glob(os.path.join(hr_sample_path, '*.png')))
            lr_paths = {}
            for qp in self.qp_list:
                lr_folder = os.path.join(self.lr_root, f"{sample_id}_640x360_30fps_{qp}")
                lr_frames = sorted(glob.glob(os.path.join(lr_folder, '*.png')))
                lr_paths[qp] = lr_frames
            num_frames = len(hr_frames)
            for i in range(num_frames):
                self.samples.append({
                    'hr': os.path.join(hr_sample_path, os.path.basename(hr_frames[i])),
                    'lr': {qp: lr_paths[qp][i] for qp in self.qp_list}
                })
        random.shuffle(self.samples)
        # self.samples = self.samples * self.repeat

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        try:
            hr_img = Image.open(sample['hr']).convert('RGB')
            hr_tensor = transforms.ToTensor()(hr_img)
        except Exception as e:
            print(f"Warning: Failed loading HR image {sample['hr']}: {e}")
            return self.__getitem__((idx + 1) % len(self))
        lr_dict = {}
        for qp, path in sample['lr'].items():
            try:
                lr_img = Image.open(path).convert('RGB')
                lr_dict[qp] = transforms.ToTensor()(lr_img)
            except Exception as e:
                print(f"Warning: Failed loading LR image {path}: {e}")
                return self.__getitem__((idx + 1) % len(self))
        return {'hr': hr_tensor, 'lr': lr_dict}


