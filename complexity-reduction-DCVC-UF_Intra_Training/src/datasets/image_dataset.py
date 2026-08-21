# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import numpy as np
import os
import random
import torch

from PIL import Image
from torch.utils.data import Dataset

from ..utils.transforms import rgb2ycbcr_np


class ImageFolder(Dataset):
    def __init__(self, root_folder_path, patch_h, patch_w, qp_num, lambdas, fixed_qp=None):
        self.root_folder_path = root_folder_path
        with open(os.path.join(root_folder_path, 'description.json')) as json_file:
            self.dataset = json.load(json_file)

        self.dataset_length = len(self.dataset)
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.qp_num = qp_num
        self.lambdas = lambdas
        # None -> sample a random QP per image (full 64-rate training). An int
        # pins every sample to that QP (single-rate training).
        assert fixed_qp is None or 0 <= fixed_qp < qp_num
        self.fixed_qp = fixed_qp

    def __getitem__(self, index):
        image_path = os.path.join(self.root_folder_path, self.dataset[index])
        img = Image.open(image_path).convert('RGB')
        width, height = img.size

        pad_height = self.patch_h - height
        pad_width = self.patch_w - width
        pad_height = max(0, pad_height)
        pad_width = max(0, pad_width)
        pad_size = ((pad_height // 2, pad_height - pad_height // 2),
                    (pad_width // 2, pad_width - pad_width // 2),
                    (0, 0),)
        padded_height = height + pad_height
        padded_width = width + pad_width
        y = random.randint(0, padded_height - self.patch_h)
        x = random.randint(0, padded_width - self.patch_w)

        if random.choice([True, False]):
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        img = np.array(img).astype(np.uint8)
        img = np.pad(img, pad_size, mode='constant')
        img = img[y:y+self.patch_h, x:x+self.patch_w, :]

        img = img.astype(np.float32) / 255.0
        img = rgb2ycbcr_np(img)
        img = img - 0.5
        img = torch.as_tensor(img, dtype=torch.float32)
        img = img.permute(2, 0, 1).contiguous()
        # img in [3, H, W]

        curr_qp = self.fixed_qp if self.fixed_qp is not None \
            else random.randint(0, self.qp_num - 1)
        curr_lambda = self.lambdas[curr_qp]
        curr_qp = torch.tensor(curr_qp, dtype=torch.int32)
        curr_lambda = torch.tensor(curr_lambda, dtype=torch.float32)
        return [img, curr_qp, curr_lambda]

    def __len__(self):
        return self.dataset_length

    def get_patch_size(self):
        return self.patch_w, self.patch_h

    def set_patch_size(self, patch_width, patch_height):
        self.patch_w = patch_width
        self.patch_h = patch_height


class ImageValFolder(Dataset):
    """Full-image validation set (no cropping / augmentation).

    Returns the whole image as YCbCr-0.5 in [-0.5, 0.5], matching the training
    ImageFolder preprocessing. Sizes vary per image, so use batch_size=1. The
    list can be a description.json (flat array) or a plain text file with one
    relative path per line (the first whitespace field is used).
    """

    def __init__(self, root_folder_path, list_path=None):
        self.root_folder_path = root_folder_path
        if list_path is None:
            list_path = os.path.join(root_folder_path, 'description.json')
        if list_path.endswith('.json'):
            with open(list_path) as f:
                self.items = json.load(f)
        else:
            with open(list_path) as f:
                self.items = [ln.strip().split()[0] for ln in f if ln.strip()]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        image_path = os.path.join(self.root_folder_path, self.items[index])
        img = Image.open(image_path).convert('RGB')
        img = np.array(img).astype(np.float32) / 255.0
        h, w = img.shape[:2]
        img = img[:h - h % 2, :w - w % 2, :]  # rgb2ycbcr_np requires even H, W
        img = rgb2ycbcr_np(img)
        img = img - 0.5
        img = torch.as_tensor(img, dtype=torch.float32).permute(2, 0, 1).contiguous()
        return img
