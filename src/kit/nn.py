import torch
import torch.nn as nn

import torchsparse
from torchsparse import nn as spnn
from torchsparse import SparseTensor
import torch.nn.functional as F


class ResNet(torch.nn.Module): 
    """Residual Network
    """  
    def __init__(self, channels, k=3):
        super().__init__()
        self.conv0 = spnn.Conv3d(channels, channels, k)
        self.conv1 = spnn.Conv3d(channels, channels, k)
        self.relu = spnn.ReLU(True)

    def forward(self, x):
        out = self.relu(self.conv0(x))
        out = self.conv1(out)
        out = self.relu(out+x)
        return out

class INR(torch.nn.Module):
    """INR
    """
    def __init__(self, channels, k=3):
        super(INR, self).__init__()
        self.conv0_0 = spnn.Conv3d(channels, channels//2, k)
        self.conv0_1 = spnn.Conv3d(channels//2, channels//2, k)
        self.conv1_0 = spnn.Conv3d(channels, channels//2, 1)
        self.conv1_1 = spnn.Conv3d(channels//2, channels//2, k)
        self.conv1_2 = spnn.Conv3d(channels//2, channels//2, 1)
        self.relu = spnn.ReLU(True)

    def forward(self, x):
        out0 = self.conv0_1(self.relu(self.conv0_0(x)))
        out1 = self.conv1_2(self.relu(self.conv1_1(self.relu(self.conv1_0(x)))))
        combined_feats = torch.cat([out0.feats, out1.feats], dim=1)
        combined_sparse = SparseTensor(coords=x.coords, feats=combined_feats)
        out = SparseTensor(coords=x.coords, feats=combined_sparse.feats + x.feats)
        return out

def make_layer(block, block_layers, channels, k=3):
    """make stacked InceptionResNet layers.
    """
    layers = []
    for i in range(block_layers):
        layers.append(block(channels=channels, k=k))
        
    return torch.nn.Sequential(*layers)

class DFA(torch.nn.Module):
    """DFA
    """
    def __init__(self, channels, k=3):
        super(DFA, self).__init__()
        self.conv1 = spnn.Conv3d(channels, channels, k)
        self.conv2 = spnn.Conv3d(channels, channels, k)
        self.relu = spnn.ReLU(True)
        self.block = make_layer(INR, 3, channels, k)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        out = self.block(x) + x
        out = self.conv2(out)
        return out



class FOG(torch.nn.Module): 
    """Fast Occupancy Converter
    """  
    def __init__(self):
        super(FOG, self).__init__()

        self.conv = spnn.Conv3d(1, 1, kernel_size=2, stride=2, bias=False)
        torch.nn.init.constant_(self.conv.kernel, 1.0)
        for param in self.conv.parameters():
            param.requires_grad = False

        self.pos_multiplier = torch.tensor([[1, 2, 4]], device='cuda')

    def pos(self, coords):
        """Assign codes (i.e., 1, 2, 4, ..., 64, 128) for each coords
        Input: coords: (N_d, 4)
        Return: pos: (N_d, 1)
        """ 
        pos = (coords[:, 1:] % 2) * self.pos_multiplier # (N_d, 3)
        pos = pos.sum(dim=-1, keepdim=True) # (N_d, 1)
        pos = (2 ** pos).float()
        return pos

    def forward(self, x):
        '''Dyadic downscaling to generate sparse occupancy code
        Input: x SparseTensor: x.coords (N_d, 4); x.feats (N_d, 1)
        Return: ds_x SparseTensor: ds_x.coords (N_{d-1}, 4); ds_x.feats (N_{d-1}, 1)
        ''' 
        x.feats = self.pos(x.coords) # (N{d-1}, 1)
        ds_x = self.conv(x) # coordinate = ds_x.C and occupancy = ds_x.F
        return ds_x


class FCG(torch.nn.Module): 
    """Fast Coordinate Converter
    """  
    def __init__(self):
        super(FCG, self).__init__()

        self.expand_coords_base = torch.tensor([
            [0, 0, 0], # -> 1 (occupancy adder)
            [1, 0, 0], # -> 2 (occupancy adder)
            [0, 1, 0], # -> 4 (occupancy adder)
            [1, 1, 0], # -> 8 (occupancy adder)
            [0, 0, 1], # -> 16 (occupancy adder)
            [1, 0, 1], # -> 32 (occupancy adder)
            [0, 1, 1], # -> 64 (occupancy adder)
            [1, 1, 1], # -> 128 (occupancy adder)
        ], device='cuda')

        self.pos = torch.arange(0, 8, device='cuda').view(1, 8)

    def forward(self, x_C, x_O, x_F=None):
        '''Upscaling according to coordinates and occupancy code
        Input: x_C: coordinates (N_d, 4)
        Input: x_O: occupancy (N_d, 1)
        Input: x_F: features (N_d, C)
        Return: x_up_C: upscaled coordinates (N_{d+1}, 4)
        Return: x_up_F: replicated features (N_{d+1}, C)
        ''' 
        # 1 to 8 expand
        expand_coords = self.expand_coords_base.repeat(x_C.shape[0], 1) # (N_d*8, 4)
        x_C_repeat = x_C.repeat(1, 8).reshape(-1, 4) # (N_d*8, 4) repeated coords
        x_C_repeat[:, 1:] = x_C_repeat[:, 1:] * 2 + expand_coords # (N_d*8, 4) expanded coords
        mask = torch.div(x_O.repeat(1, 8) % (2**(self.pos+1)), 2**self.pos, rounding_mode='floor').reshape(-1) # (N_d*8, 1) mask for pruning
        mask = (mask == 1) # (N_d*8, 1) mask for pruning
        x_up_C = x_C_repeat[mask].int() # (N_{d+1}, 4) upscaled coords
        if x_F is None:
            return x_up_C
        else:
            C = x_F.shape[1]
            x_F = x_F.repeat(1, 8).reshape(-1, C) # (N_d*8, C)
            x_up_F = x_F[mask]
            return x_up_C, x_up_F


class FCG_from_indices(torch.nn.Module):
    """
    Fast Coordinate Generator from parent coordinates and child indices.
    This is a variant of FCG used for lossy decoding. In lossy decoding, we do not 
    have a complete occupancy code, but rather a list of child voxel indices to generate.
    """
    def __init__(self):
        super(FCG_from_indices, self).__init__()

        # Use register_buffer so that when model.to(device) is called, 
        # this tensor is automatically moved to the correct device
        self.register_buffer('expand_coords_base', torch.tensor([
            [0, 0, 0], # -> child_index 0
            [1, 0, 0], # -> child_index 1
            [0, 1, 0], # -> child_index 2
            [1, 1, 0], # -> child_index 3
            [0, 0, 1], # -> child_index 4
            [1, 0, 1], # -> child_index 5
            [0, 1, 1], # -> child_index 6
            [1, 1, 1], # -> child_index 7
        ], dtype=torch.int32))

    def forward(self, parent_coords, child_indices, parent_features=None):
        """
        Coordinate upsampling based on parent coordinates and child indices.

        Inputs:
            parent_coords: Coordinates of parent voxels (M, 4)
            child_indices: Local child indices within their parent voxel (0-7) (M,)
            parent_features: Features of parent voxels (M, C), optional.
        Returns:
            child_coords: Upsampled coordinates (M, 4)
            child_features: Passed features (M, C), if parent_features are provided.
        """
        # 1. Get the corresponding coordinate offsets for each child index.
        offsets = self.expand_coords_base[child_indices] # (M, 3)

        # 2. Calculate child coordinates.
        # parent_coord * 2 + offset
        child_xyz = parent_coords[:, 1:] * 2 + offsets # (M, 3)

        # 3. Combine with batch indices.
        batch_index = parent_coords[:, 0:1] # (M, 1)
        child_coords = torch.cat([batch_index, child_xyz], dim=1)

        if parent_features is None:
            return child_coords.int()
        else:
            # Parent features are directly mapped (passed) to their children.
            return child_coords.int(), parent_features


class TargetEmbedding(torch.nn.Module): 
    """Target Embedding
    """  
    def __init__(self, channels):
        super(TargetEmbedding, self).__init__()
        self.target_res_embedding = nn.Embedding(8, channels)

    def forward(self, x_up_F, x_up_C):
        '''Embed x.F from x.C to x_up_C
        Input: x_up_F Feats (N_{d+1}, dim)
        Input: x_up_C Coords (N_{d+1}, 4)
        Return x_up SparseTensor (N_{d+1}, dim)
        ''' 
        coords_delta = x_up_C[:, 1:] % 2
        coords_idx = coords_delta[:, 0] + coords_delta[:, 1]*2 + coords_delta[:, 2]*4
        x_up_F = x_up_F + self.target_res_embedding(coords_idx.int()) # (B*Nt, C)
        return x_up_F