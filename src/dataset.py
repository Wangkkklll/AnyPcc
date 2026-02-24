import torch
from torchsparse import SparseTensor
import kit.io as io
from kit.io import kdtree_partition
import random
import numpy as np
import os
from glob import glob
import json
import logging
import torch.utils.data

class LazyPointCloudDataset:
    """
    Base class for lazy-loading point cloud datasets.
    """
    def __init__(self, file_path_ls, **kwargs):
        self.file_path_ls = file_path_ls
        self.kwargs = kwargs

    def __len__(self):
        return len(self.file_path_ls)

    def _load_and_process_file(self, file_path):
        """
        Loads and processes a single point cloud file.
        Can be overridden by subclasses to implement custom logic.
        """
        if file_path.endswith('.h5'):
            xyz = io.read_point_cloud_h5(file_path)
        else:
            xyz = io.read_point_cloud(file_path)
        
        # Ensure only the first 3 columns (x, y, z) are used, ignoring extra features
        if xyz.shape[1] > 3:
            xyz = xyz[:, :3]

        posQ = self.kwargs.get('posQ', 1)
        is_pre_quantized = self.kwargs.get('is_pre_quantized', False)
        use_augmentation = self.kwargs.get('use_augmentation', False)
        shift_range = self.kwargs.get('shift_range', 1)
        use_patch = self.kwargs.get('use_patch', False)
        max_num = self.kwargs.get('max_num', 200000)
        preprocess_scale = self.kwargs.get('preprocess_scale', 1.0)
        preprocess_shift = self.kwargs.get('preprocess_shift', 0.0)

        
        # Extract a random patch if patching is enabled and the point cloud exceeds the max limit
        if use_patch and len(xyz) > max_num:
            parts = kdtree_partition(xyz, max_num=max_num)
            xyz = random.sample(parts, 1)[0]
            
        xyz = torch.tensor(xyz, dtype=torch.float)
        
        # Apply dataset-specific scaling and shifting
        if not is_pre_quantized:
            xyz = xyz * preprocess_scale + preprocess_shift
        
        feats = torch.ones((xyz.shape[0], 1), dtype=torch.float)

        # Apply data augmentation via random spatial scaling
        if use_augmentation:
            # Randomly choose between 1.0 (no scaling) or the specified shift_range scale
            scale = random.choice([1.0, shift_range]) 
            xyz_quantized = torch.round(xyz / posQ / scale).int()
        else:
            xyz_quantized = torch.round(xyz / posQ).int()
        
        coords = xyz_quantized.int()
        input_tensor = SparseTensor(coords=coords, feats=feats)
        
        return {"input": input_tensor, "file_path": file_path}

    def __getitem__(self, idx):
        file_path = self.file_path_ls[idx]
        return self._load_and_process_file(file_path)


class UnifiedPCDataset(LazyPointCloudDataset):
    """
    A unified point cloud dataset class integrating the following features:
    - Reads from various file formats (.h5, etc.).
    - Optional pre-quantization.
    - Data augmentation via random scaling.
    - Large point cloud patching via k-d tree partitioning.
    - Dataset-specific preprocessing (scaling and shifting).
    - Lazy loading to reduce memory footprint.
    """
    def __init__(self, 
                 file_path_ls, 
                 posQ=1, 
                 is_pre_quantized=False, 
                 use_augmentation=False, 
                 shift_range=1024,
                 use_patch=False, 
                 max_num=200000,
                 preprocess_scale=1.0,
                 preprocess_shift=0.0):
        
        super().__init__(
            file_path_ls,
            posQ=posQ,
            is_pre_quantized=is_pre_quantized,
            use_augmentation=use_augmentation,
            shift_range=shift_range,
            use_patch=use_patch,
            max_num=max_num,
            preprocess_scale=preprocess_scale,
            preprocess_shift=preprocess_shift
        )