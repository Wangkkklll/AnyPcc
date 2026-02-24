import os
import time
import random
import argparse

import numpy as np
from glob import glob
from tqdm import tqdm

import torch
import torchac

from torchsparse import SparseTensor
from torchsparse.nn import functional as F

import kit.io as io
import kit.op as op
from kit.op import z_order_sort

import pandas as pd
import logging
import deepCABAC
from torch.optim.lr_scheduler import StepLR

# Threshold to split large point clouds to avoid OOM
limited_num = 1500000 

random.seed(1)
np.random.seed(1)
device = 'cuda'


@torch.no_grad()
def whole_step_forward(net, x):
    """
    Initial forward pass to compute and cache intermediate features and ground truths.
    """
    with torch.no_grad():
        N = x.coords.shape[0]
        
        # Get sparse occupancy code list via FOG downsampling
        data_ls = []
        while True:
            x = net.fog(x)
            data_ls.append((x.coords.clone(), x.feats.clone())) 
            if x.coords.shape[0] < 64 * 16:
                break
        data_ls = data_ls[::-1]

        total_bits = 0
        MLP_dairy = {}
        GT_dairy = {}
        group1_bits = 0
        group2_bits = 0
        
        for depth in range(len(data_ls) - 1):
            x_C, x_O = data_ls[depth]
            gt_x_up_C, gt_x_up_O = data_ls[depth + 1]
            gt_x_up_C, gt_x_up_O = z_order_sort(gt_x_up_C, gt_x_up_O)

            # Feature extraction
            x_F = net.prior_embedding(x_O.int()).view(-1, net.channels)
            x = SparseTensor(coords=x_C, feats=x_F)
            x = net.prior_resnet(x)

            x_up_C, x_up_F = net.fcg(x_C, x_O, x.feats)
            x_up_C, x_up_F = z_order_sort(x_up_C, x_up_F)

            x_up_F = net.target_embedding(x_up_F, x_up_C)
            x_up = SparseTensor(coords=x_up_C, feats=x_up_F)
            x_up = net.target_resnet(x_up)

            # 3D checkerboard grouping
            group1_mask, group2_mask = net.get_3d_checkerboard_groups(gt_x_up_C)
            MLP_dairy[depth] = []
            GT_dairy[depth] = []
            
            # Group 1 encoding (White nodes, basic spatial convolution)
            if group1_mask.sum() > 0:
                group1_coords = gt_x_up_C[group1_mask]
                group1_features = x_up.feats[group1_mask]
                group1_occupancy = gt_x_up_O[group1_mask]

                group1_sparse = SparseTensor(coords=group1_coords, feats=group1_features)

                gt_s0 = torch.remainder(group1_occupancy, 16)
                gt_s1 = torch.div(group1_occupancy, 16, rounding_mode='floor')

                GT_dairy[depth].append(gt_s0)
                GT_dairy[depth].append(gt_s1)

                # Stage 1 prediction
                group1_s0_sparse = net.group1_spatial_conv_s0(group1_sparse)
                MLP_dairy[depth].append(group1_s0_sparse.feats.detach())
                
                prob_s0 = net.group1_pred_head_s0(group1_s0_sparse.feats)
                selected_prob_s0 = prob_s0.gather(1, gt_s0.long())
                bits_s0 = torch.sum(torch.clamp(-torch.log2(selected_prob_s0 + 1e-10), 0, 50))

                # Stage 2 prediction (conditioned on Stage 1 results)
                s1_emb = net.group1_pred_head_s1_emb(gt_s0[:, 0].long())
                group1_s1_F = group1_sparse.feats + s1_emb
                group1_s1_sparse = SparseTensor(coords=group1_coords, feats=group1_s1_F)
                group1_s1_sparse = net.group1_spatial_conv_s1(group1_s1_sparse)
                
                MLP_dairy[depth].append(group1_s1_sparse.feats.detach())
                prob_s1 = net.group1_pred_head_s1(group1_s1_sparse.feats)
                selected_prob_s1 = prob_s1.gather(1, gt_s1.long())
                bits_s1 = torch.sum(torch.clamp(-torch.log2(selected_prob_s1 + 1e-10), 0, 50))

                group1_bits += (bits_s0 + bits_s1)

            # Group 2 encoding (Black nodes, enhanced spatial convolution + Group 1 info)
            if group2_mask.sum() > 0 and group1_mask.sum() > 0:
                group2_coords = gt_x_up_C[group2_mask]
                group2_occupancy = gt_x_up_O[group2_mask]

                enhanced_features = net.aggregate_neighbor_features_efficient(
                    gt_x_up_C, x_up.feats, group1_mask, group2_mask, gt_x_up_O
                )

                group2_sparse = SparseTensor(coords=group2_coords, feats=enhanced_features)

                gt_s0 = torch.remainder(group2_occupancy, 16)
                gt_s1 = torch.div(group2_occupancy, 16, rounding_mode='floor')
                
                GT_dairy[depth].append(gt_s0)
                GT_dairy[depth].append(gt_s1)

                # Stage 1 prediction
                group2_s0_sparse = net.group2_spatial_conv_s0(group2_sparse)
                MLP_dairy[depth].append(group2_s0_sparse.feats.detach())
                
                prob_s0 = net.group2_pred_head_s0(group2_s0_sparse.feats)
                selected_prob_s0 = prob_s0.gather(1, gt_s0.long())
                bits_s0 = torch.sum(torch.clamp(-torch.log2(selected_prob_s0 + 1e-10), 0, 50))

                # Stage 2 prediction
                s1_emb = net.group2_pred_head_s1_emb(gt_s0[:, 0].long())
                group2_s1_F = group2_sparse.feats + s1_emb
                group2_s1_sparse = SparseTensor(coords=group2_coords, feats=group2_s1_F)
                group2_s1_sparse = net.group2_spatial_conv_s1(group2_s1_sparse)
                
                MLP_dairy[depth].append(group2_s1_sparse.feats.detach())
                prob_s1 = net.group2_pred_head_s1(group2_s1_sparse.feats)
                selected_prob_s1 = prob_s1.gather(1, gt_s1.long())
                bits_s1 = torch.sum(torch.clamp(-torch.log2(selected_prob_s1 + 1e-10), 0, 50))

                group2_bits += (bits_s0 + bits_s1)

            elif group2_mask.sum() > 0 and group1_mask.sum() == 0:
                # Fallback to standard encoding when Group 1 is empty
                group2_coords = gt_x_up_C[group2_mask]
                group2_features = x_up.feats[group2_mask]
                group2_occupancy = gt_x_up_O[group2_mask]

                group2_sparse = SparseTensor(coords=group2_coords, feats=group2_features)

                gt_s0 = torch.remainder(group2_occupancy, 16)
                gt_s1 = torch.div(group2_occupancy, 16, rounding_mode='floor')

                GT_dairy[depth].append(gt_s0)
                GT_dairy[depth].append(gt_s1)
                
                # Stage 1 prediction
                group2_s0_sparse = net.group2_spatial_conv_s0(group2_sparse)
                MLP_dairy[depth].append(group2_s0_sparse.feats.detach())
                
                prob_s0 = net.group2_pred_head_s0(group2_s0_sparse.feats)
                selected_prob_s0 = prob_s0.gather(1, gt_s0.long())
                bits_s0 = torch.sum(torch.clamp(-torch.log2(selected_prob_s0 + 1e-10), 0, 50))

                # Stage 2 prediction
                s1_emb = net.group2_pred_head_s1_emb(gt_s0[:, 0].long())
                group2_s1_F = group2_sparse.feats + s1_emb
                group2_s1_sparse = SparseTensor(coords=group2_coords, feats=group2_s1_F)
                group2_s1_sparse = net.group2_spatial_conv_s1(group2_s1_sparse)
                
                MLP_dairy[depth].append(group2_s1_sparse.feats.detach())
                prob_s1 = net.group2_pred_head_s1(group2_s1_sparse.feats)
                selected_prob_s1 = prob_s1.gather(1, gt_s1.long())
                bits_s1 = torch.sum(torch.clamp(-torch.log2(selected_prob_s1 + 1e-10), 0, 50))

                group2_bits += (bits_s0 + bits_s1)

            total_bits = group1_bits + group2_bits
            bpp = total_bits / N
            
    return bpp, MLP_dairy, GT_dairy


def only_mlp_forward(net, MLP_dairy, GT_dairy):
    """
    Forward pass strictly through the prediction heads to compute gradients efficiently.
    """
    total_bits = 0

    for depth in MLP_dairy:
        MLP_ls = MLP_dairy[depth]
        gt_x_up_O_s0, gt_x_up_O_s1, gt_x_up_O_s2, gt_x_up_O_s3 = GT_dairy[depth]

        # Ensure input features require gradients
        feats_s0 = MLP_ls[0].requires_grad_(True).clone()
        feats_s1 = MLP_ls[1].requires_grad_(True).clone()
        feats_s2 = MLP_ls[2].requires_grad_(True).clone()
        feats_s3 = MLP_ls[3].requires_grad_(True).clone()

        # Stage 1
        x_up_O_prob_s0 = net.group1_pred_head_s0(feats_s0)
        x_up_O_prob_s0 = x_up_O_prob_s0.gather(1, gt_x_up_O_s0.long())

        # Stage 2
        x_up_O_prob_s1 = net.group1_pred_head_s1(feats_s1)
        x_up_O_prob_s1 = x_up_O_prob_s1.gather(1, gt_x_up_O_s1.long())

        # Stage 3
        x_up_O_prob_s2 = net.group2_pred_head_s0(feats_s2)
        x_up_O_prob_s2 = x_up_O_prob_s2.gather(1, gt_x_up_O_s2.long())

        # Stage 4
        x_up_O_prob_s3 = net.group2_pred_head_s1(feats_s3)
        x_up_O_prob_s3 = x_up_O_prob_s3.gather(1, gt_x_up_O_s3.long())

        # Accumulate loss (bits)
        total_bits += torch.sum(torch.clamp(-1.0 * torch.log2(x_up_O_prob_s0 + 1e-10), 0, 50))
        total_bits += torch.sum(torch.clamp(-1.0 * torch.log2(x_up_O_prob_s1 + 1e-10), 0, 50))
        total_bits += torch.sum(torch.clamp(-1.0 * torch.log2(x_up_O_prob_s2 + 1e-10), 0, 50))
        total_bits += torch.sum(torch.clamp(-1.0 * torch.log2(x_up_O_prob_s3 + 1e-10), 0, 50))
            
    return total_bits


@torch.no_grad()
def compress_net(lossless_net, compress_ls, save_path='weights.bin'):
    """
    Compress the fine-tuned model parameters using deepCABAC.
    """
    encoder = deepCABAC.Encoder()
    
    interv = 0.1
    stepsize = 2**(-0.5 * 16)
    stepsize_other = 2**(-0.5 * 20)
    _lambda = 0.
    
    compress_dict = {}

    for full_param_name, param in lossless_net.named_parameters():
        # Check if the parameter belongs to the target fine-tuning modules
        in_postrain_module = any(full_param_name.startswith(module_name) for module_name in compress_ls)
        
        # Check if the parameter is in the target layers (e.g., '.2.' or '.4.')
        is_target_layer = '.2.' in full_param_name or '.4.' in full_param_name or '.0.' in full_param_name

        if in_postrain_module and is_target_layer:
            compress_dict[full_param_name] = param.clone()
            param_np = param.detach().cpu().numpy()
            
            if '.weight' in full_param_name:
                encoder.encodeWeightsRD(param_np, interv, stepsize, _lambda)
            elif '.kernel' in full_param_name: 
                # For sparse convolution
                encoder.encodeWeightsRD(param_np.reshape(-1), interv, stepsize, _lambda)
            else:
                encoder.encodeWeightsRD(param_np, interv, stepsize_other, _lambda)

    stream = encoder.finish().tobytes()

    with open(save_path, 'wb') as f:
        f.write(stream)

    compressed_size = op.get_file_size_in_bits(save_path)

    # Decoding stage to verify reconstruction and load weights back
    reconstructed_model = lossless_net
    decoder = deepCABAC.Decoder()

    with open(save_path, 'rb') as f:
        stream = f.read()

    decoder.getStream(np.frombuffer(stream, dtype=np.uint8))
    state_dict = reconstructed_model.state_dict()

    for full_param_name, param in lossless_net.named_parameters():
        in_postrain_module = any(full_param_name.startswith(module_name) for module_name in compress_ls)
        is_target_layer = '.2.' in full_param_name or '.4.' in full_param_name or '.0.' in full_param_name

        if in_postrain_module and is_target_layer:
            if '.kernel' in full_param_name:  
                param = decoder.decodeWeights()
                state_dict[full_param_name] = torch.tensor(param.reshape(8, 32, -1)).cuda()
            else:
                param = decoder.decodeWeights()
                state_dict[full_param_name] = torch.tensor(param).cuda()

    decoder.finish()
    reconstructed_model.load_state_dict(state_dict)

    return reconstructed_model, compressed_size


def decompress_net(lossless_net, save_path):
    """
    Decompress model parameters from a bin file using deepCABAC.
    """
    reconstructed_model = lossless_net
    decoder = deepCABAC.Decoder()

    with open(save_path, 'rb') as f:
        stream = f.read()

    decoder.getStream(np.frombuffer(stream, dtype=np.uint8))
    state_dict = reconstructed_model.state_dict()

    compress_ls = ["group1_pred_head_s1", "group1_pred_head_s0", "group2_pred_head_s1", "group2_pred_head_s0"]
    
    for full_param_name, param in lossless_net.named_parameters():
        in_postrain_module = any(full_param_name.startswith(module_name) for module_name in compress_ls)
        is_target_layer = '.2.' in full_param_name or '.4.' in full_param_name or '.0.' in full_param_name

        if in_postrain_module and is_target_layer:
            if '.kernel' in full_param_name:  
                param = decoder.decodeWeights()
                state_dict[full_param_name] = torch.tensor(param.reshape(8, 32, -1)).cuda()
            else:
                param = decoder.decodeWeights()
                state_dict[full_param_name] = torch.tensor(param).cuda()

    decoder.finish()
    reconstructed_model.load_state_dict(state_dict)

    return reconstructed_model


def split_points_by_x(x, max_points=limited_num):
    """
    Split the point cloud along the x-axis to ensure the number of points 
    per chunk does not exceed the specified limit.
    """
    # Assuming x is the first spatial dimension (index 1 in coords)
    x_coords = x.coords[:, 1]
    coords = x.coords
    
    total_points = coords.shape[0]
    num_chunks = (total_points + max_points - 1) // max_points
    
    if num_chunks <= 1:
        return [x]
    
    sorted_indices = torch.argsort(x_coords)
    sorted_coords = coords[sorted_indices]
    
    chunks = []
    for i in range(num_chunks):
        start_idx = i * max_points
        end_idx = min((i + 1) * max_points, total_points)
        chunk = sorted_coords[start_idx:end_idx]
        chunk_x = SparseTensor(coords=chunk, feats=torch.ones(chunk.shape[0], 1).float())
        chunks.append(chunk_x)
    
    return chunks


def init_net(net, x, pre_ckpt_path, postrain_ls, lr, save_path='weights.bin', max_epoch=800, tune=True):
    """
    Main entry point for full fine-tuning.
    """
    # Load model parameters from the pre-trained checkpoint
    net.load_state_dict(torch.load(pre_ckpt_path))
    
    if not tune:
        return net, 0.0, 0.0
        
    st_time = time.time()
    
    with torch.no_grad():
        N = x.coords.shape[0]
        chunks = []
        if N > limited_num:
            chunks = split_points_by_x(x)
        else:
            chunks = [x]

        # Perform initial forward pass to get feature maps and ground truth
        MLP_ls = []
        GT_ls = []
        for chunk_x in chunks:
            bpp, MLP_dairy, GT_dairy = whole_step_forward(net, chunk_x)
            MLP_ls.append(MLP_dairy)
            GT_ls.append(GT_dairy)
    
    # Freeze all parameters initially
    for name, param in net.named_parameters():
        param.requires_grad = False
    
    # Unfreeze and train parameters in specified modules
    trainable_params = []
    for full_param_name, param in net.named_parameters():
        in_postrain_module = any(full_param_name.startswith(module_name) for module_name in postrain_ls)
        is_target_layer = '.2.' in full_param_name or '.4.' in full_param_name or '.0.' in full_param_name

        if in_postrain_module and is_target_layer:
            param.requires_grad = True
            trainable_params.append(param)
    
    # Create optimizer for trainable parameters
    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    l1_lambda = 0.0004

    # Training loop
    t_start = time.time()
    
    with torch.enable_grad():
        for epoch in range(max_epoch):
            for idx in range(len(GT_ls)):
                MLP_dairy = MLP_ls[idx]
                GT_dairy = GT_ls[idx]
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass (only through MLP heads)
                total_bits = only_mlp_forward(net, MLP_dairy, GT_dairy)
                bpp = total_bits / N
                
                # Calculate L1 regularization loss
                l1_loss = 0
                for param in trainable_params:
                    l1_loss += torch.sum(torch.abs(param)) 
                
                if epoch < 0:
                    all_loss = bpp
                else:
                    all_loss = bpp + l1_lambda * l1_loss

                # Backpropagation
                all_loss.backward()
                
                # Update parameters
                optimizer.step()

            # Logging progress
            if epoch % 50 == 0:
                elapsed = time.time() - t_start
                print(f"Epoch {epoch}, BPP: {bpp.item():.6f}, L1-loss: {l1_loss.item():.6f}, Time: {elapsed:.2f}s")

    # Encode compressed weights after training
    reconstructed_model, compressed_size = compress_net(net, postrain_ls, save_path)
    
    return reconstructed_model, compressed_size, time.time() - st_time