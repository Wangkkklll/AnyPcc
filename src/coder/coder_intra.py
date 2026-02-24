import os
import time
import random
import argparse
from collections import Counter

import numpy as np
from glob import glob
from tqdm import tqdm

import torch
import torchac
import torch.nn as nn

from torchsparse import SparseTensor
from torchsparse.nn import functional as F

import kit.io as io
import kit.op as op
from kit.op import z_order_sort, calc_morton_code, _top2_avoid_zero

from kit.nn import FCG_from_indices

import pandas as pd
import sys
import gc
import copy
import tempfile

from model import UCM_Context_Model as Lossless_Model
from model import UCM_Context_Model_1stage as Lossy_Model


"""
Current Coder supports: Lossless Compression / Lossy Compression / Full Fine-tuning
"""

class AnyPcc_CoderIntra(nn.Module):
    def __init__(self, model_path, device='cuda:0', channels=32, kernel_size=3, 
                 posQ=16, preprocess_scale=1.0, preprocess_shift=0.0, 
                 is_data_pre_quantized=False,
                # Lossy compression parameters
                lossy_model_path=None, lossy_level=0, 
                no_lossy_net=False, fixed_k=False,
                # Full fine-tuning parameters
                tune=False, postrain_lr=0.001, postrain_ls=None, epoch=800,
                tune_threshold=20000):
        super(AnyPcc_CoderIntra, self).__init__()
        self.device = device
        self.posQ = posQ
        self.preprocess_scale = preprocess_scale
        self.preprocess_shift = preprocess_shift
        self.is_data_pre_quantized = is_data_pre_quantized
        self._model_path = model_path  
        
        self.lossy_level = lossy_level
        self.no_lossy_net = no_lossy_net
        self.fixed_k = fixed_k  # Whether to decode a fixed 'k' number of points
        
        # Initialize FCG_from_indices if using the fixed k strategy
        if self.fixed_k:
            if FCG_from_indices is None:
                raise ImportError("FCG_from_indices import failed, required for fixed_k=True")
            self.fcg_from_indices = FCG_from_indices().to(device)
        else:
            self.fcg_from_indices = None
        
        self.tune = tune
        self.postrain_lr = postrain_lr
        self.postrain_ls = postrain_ls if postrain_ls is not None else []
        self.epoch = epoch
        self.tune_threshold = tune_threshold
        
        # Set torchsparse config
        conv_config = F.conv_config.get_default_conv_config()
        conv_config.kmap_mode = "hashmap"
        F.conv_config.set_global_conv_config(conv_config)
        
        # Load lossless model
        self.model = Lossless_Model(channels=channels, kernel_size=kernel_size, device=device)
        if model_path and os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location=device)
            if 'model_state_dict' in ckpt:
                ckpt = ckpt['model_state_dict']
            self.model.load_state_dict(ckpt, strict=False)
        self.model.to(device)
        self.model.eval()

        # Load lossy model (if provided)
        self.lossy_model = None
        if lossy_model_path and os.path.exists(lossy_model_path) and lossy_level > 0:
            self.lossy_model = Lossy_Model(channels=channels, kernel_size=kernel_size, device=device)
            lossy_ckpt = torch.load(lossy_model_path, map_location=device)
            if 'model_state_dict' in lossy_ckpt:
                lossy_ckpt = lossy_ckpt['model_state_dict']
            self.lossy_model.load_state_dict(lossy_ckpt, strict=False)
            self.lossy_model.to(device)
            self.lossy_model.eval()

        # Warm-up
        random_coords = torch.randint(low=0, high=2048, size=(2048, 3)).int().to(device)
        self.model(SparseTensor(coords=torch.cat((random_coords[:, 0:1]*0, random_coords), dim=-1),
                    feats=torch.ones((2048, 1))).to(device))

    def compress(self, xyz, return_metadata=False, lossy_level=None):
        """
        Compress point cloud data.
        
        Args:
            xyz: numpy array or torch tensor, shape (N, 3) point cloud coordinates
            return_metadata: bool, whether to return metadata (used for testing)
            lossy_level: int, number of lossy levels. Uses self.lossy_level if None. 0 means lossless.
        
        Returns:
            byte_stream: compressed byte stream (includes metadata and compressed data)
            metadata/enc_time: dict with posQ, base_x_coords, base_x_feats, enc_time if return_metadata=True, 
                               otherwise returns encoding time (seconds).
        """
        with torch.no_grad():
            current_lossy_level = lossy_level if lossy_level is not None else self.lossy_level
            
            if current_lossy_level > 0 and self.lossy_model is None and not self.no_lossy_net:
                raise ValueError("Lossy mode requires lossy_model, but path is missing or model not loaded")
            
            if isinstance(xyz, np.ndarray):
                if self.is_data_pre_quantized:
                    xyz = torch.tensor(xyz, device=self.device)
                else:
                    xyz = torch.tensor(xyz / self.preprocess_scale, device=self.device)
            else:
                xyz = xyz.to(self.device)
                if not self.is_data_pre_quantized:
                    xyz = xyz / self.preprocess_scale
            
            xyz = torch.round((xyz + self.preprocess_shift) / self.posQ).int()
            N = xyz.shape[0]
            
            xyz = torch.cat((xyz[:,0:1]*0, xyz), dim=-1).int()
            feats = torch.ones((xyz.shape[0], 1), dtype=torch.float, device=self.device)
            x = SparseTensor(coords=xyz, feats=feats).to(self.device)
            
            if self.device.startswith('cuda'):
                torch.cuda.synchronize()
            enc_time_start = time.time()
            
            # Fine-tuning logic
            net_bits = 0
            tune_time = 0
            use_finetuned_model = False
            tuned_params_bytes = b''
            
            if len(self.postrain_ls) == 0:
                postrain_ls = [f'group{group}_pred_head_s{stage}' for group in [1, 2] for stage in range(2)]
            else:
                postrain_ls = self.postrain_ls

            if not self.tune or N < self.tune_threshold:
                # Skip fine-tuning: load original pretrained weights to prevent contamination
                if self._model_path and os.path.exists(self._model_path):
                    ckpt = torch.load(self._model_path, map_location=self.device)
                    if 'model_state_dict' in ckpt:
                        ckpt = ckpt['model_state_dict']
                    self.model.load_state_dict(ckpt)
                self.model.eval()
            else:
                # Execute full fine-tuning
                ckpt = torch.load(self._model_path, map_location=self.device)
                if 'model_state_dict' in ckpt:
                    ckpt = ckpt['model_state_dict']
                self.model.load_state_dict(ckpt)
                self.model.eval()

                with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as tmp_file:
                    net_save_path = tmp_file.name
                from kit.pos_train import init_net

                # Direct model fine-tuning; init_net handles initial weight overrides
                self.model, net_bits, tune_time = init_net(
                    net=self.model, x=x, pre_ckpt_path=self._model_path,
                    postrain_ls=postrain_ls, lr=self.postrain_lr,
                    save_path=net_save_path, max_epoch=self.epoch, tune=True
                )
                
                if os.path.exists(net_save_path) and net_bits > 0:
                    with open(net_save_path, 'rb') as f:
                        tuned_params_bytes = f.read()
                
                use_finetuned_model = True
                print(f"Fine-tuning completed (Time: {tune_time:.2f}s, Param size: {net_bits/8:.2f} bytes)")
                
            # Preprocessing: Downsample via fog 
            data_ls = []
            while True:
                x = self.model.fog(x)
                data_ls.append((x.coords.clone(), x.feats.clone()))
                if x.coords.shape[0] < 64:
                    break
            data_ls = data_ls[::-1]
            
            num_layers = len(data_ls) - 1
            byte_stream_ls = []

            # Neural network inference and encoding
            for depth in range(num_layers):
                if self.no_lossy_net and depth >= num_layers - current_lossy_level:
                    break
                
                is_lossy_layer = depth >= num_layers - current_lossy_level
                use_lossless_blocks = not is_lossy_layer
                net_for_blocks = self.model if use_lossless_blocks else self.lossy_model
                is_pure_lossy = is_lossy_layer
                
                x_C, x_O = data_ls[depth]
                x_C, x_O = z_order_sort(x_C, x_O)
                gt_x_up_C, gt_x_up_O = data_ls[depth+1]
                gt_x_up_C, gt_x_up_O = z_order_sort(gt_x_up_C, gt_x_up_O)
                
                # Backbone Forward
                x_F = net_for_blocks.prior_embedding(x_O.int()).view(-1, net_for_blocks.channels)
                x = SparseTensor(coords=x_C, feats=x_F)
                x = net_for_blocks.prior_resnet(x)
                
                # FCG then Sort
                x_up_C, x_up_F = net_for_blocks.fcg(x_C, x_O, x.feats)
                x_up_C, x_up_F = z_order_sort(x_up_C, x_up_F)
                
                x_up_F = net_for_blocks.target_embedding(x_up_F, x_up_C)
                x_up = SparseTensor(coords=x_up_C, feats=x_up_F)
                x_up = net_for_blocks.target_resnet(x_up)
                
                if not torch.equal(x_up.coords, x_up_C):
                    x_up_C_out, x_up_F_out = z_order_sort(x_up.coords, x_up.feats)
                    x_up = SparseTensor(coords=x_up_C_out, feats=x_up_F_out)
                    x_up_C = x_up_C_out
                
                # Process pure lossy layer
                if is_pure_lossy:
                    # Pure lossy layer: only encode 'k' (point count)
                    if self.fixed_k:
                        # Fixed k strategy: k is the true point count, obtained via sum of popcounts
                        shifters = torch.arange(0, 8, device=gt_x_up_O.device, dtype=torch.int32).view(1, -1)
                        k = ((gt_x_up_O.int() >> shifters) & 1).sum().item()
                    else:
                        # Unfixed k strategy: k is the number of voxels
                        k = gt_x_up_C.shape[0]
                    k_bytes = np.array(k, dtype=np.uint32).tobytes()
                    byte_stream_ls.append(k_bytes)
                    continue
                
                # 3D checkerboard grouping (using gt_x_up_C)
                group1_mask, group2_mask = self.model.get_3d_checkerboard_groups(gt_x_up_C)
                
                # Group 1 encoding (2 stages: s0 and s1)
                if group1_mask.sum() > 0:
                    group1_coords = gt_x_up_C[group1_mask]
                    group1_features = x_up.feats[group1_mask]
                    group1_occupancy = gt_x_up_O[group1_mask].long()
                    
                    # Split occupancy into s0 (lower 4 bits) and s1 (higher 4 bits)
                    gt_s0 = torch.remainder(group1_occupancy, 16).long()
                    gt_s1 = torch.div(group1_occupancy, 16, rounding_mode='floor').long()
                    
                    # Stage 0
                    group1_sparse = SparseTensor(coords=group1_coords, feats=group1_features)
                    group1_s0_sparse = self.model.group1_spatial_conv_s0(group1_sparse)
                    prob_s0 = self.model.group1_pred_head_s0(group1_s0_sparse.feats)
                    
                    cdf_s0 = torch.cat((prob_s0[:, 0:1]*0, prob_s0.cumsum(dim=-1)), dim=-1)
                    cdf_norm_s0 = op._convert_to_int_and_normalize(torch.clamp(cdf_s0, 0, 1), True).cpu()
                    byte_stream_s0 = torchac.encode_int16_normalized_cdf(
                        cdf_norm_s0, gt_s0.squeeze(-1).to(torch.int16).cpu()
                    )
                    byte_stream_ls.append(byte_stream_s0)
                    
                    # Stage 1 (conditioned on s0 results)
                    s1_emb = self.model.group1_pred_head_s1_emb(gt_s0.squeeze(-1))
                    group1_s1_F = group1_features + s1_emb
                    group1_s1_sparse = SparseTensor(coords=group1_coords, feats=group1_s1_F)
                    group1_s1_sparse = self.model.group1_spatial_conv_s1(group1_s1_sparse)
                    prob_s1 = self.model.group1_pred_head_s1(group1_s1_sparse.feats)
                    
                    cdf_s1 = torch.cat((prob_s1[:, 0:1]*0, prob_s1.cumsum(dim=-1)), dim=-1)
                    cdf_norm_s1 = op._convert_to_int_and_normalize(torch.clamp(cdf_s1, 0, 1), True).cpu()
                    byte_stream_s1 = torchac.encode_int16_normalized_cdf(
                        cdf_norm_s1, gt_s1.squeeze(-1).to(torch.int16).cpu()
                    )
                    byte_stream_ls.append(byte_stream_s1)
                else:
                    byte_stream_ls.extend([b''] * 2)
                
                # Group 2 encoding (2 stages, utilizing Group 1 info)
                if group2_mask.sum() > 0:
                    group2_occupancy_raw = gt_x_up_O[group2_mask]
                    
                    if group1_mask.sum() > 0:
                        enhanced_features = self.model.aggregate_neighbor_features_efficient(
                            gt_x_up_C, x_up.feats, group1_mask, group2_mask, gt_x_up_O
                        )
                        group2_coords = gt_x_up_C[group2_mask]
                        group2_occupancy = group2_occupancy_raw.long()
                    else:
                        group2_coords = gt_x_up_C[group2_mask]
                        enhanced_features = x_up.feats[group2_mask]
                        group2_occupancy = group2_occupancy_raw.long()
                    
                    gt_s0 = torch.remainder(group2_occupancy, 16).long()
                    gt_s1 = torch.div(group2_occupancy, 16, rounding_mode='floor').long()
                    
                    # Stage 0
                    group2_sparse = SparseTensor(coords=group2_coords, feats=enhanced_features)
                    group2_s0_sparse = self.model.group2_spatial_conv_s0(group2_sparse)
                    prob_s0 = self.model.group2_pred_head_s0(group2_s0_sparse.feats)
                    
                    cdf_s0 = torch.cat((prob_s0[:, 0:1]*0, prob_s0.cumsum(dim=-1)), dim=-1)
                    cdf_norm_s0 = op._convert_to_int_and_normalize(torch.clamp(cdf_s0, 0, 1), True).cpu()
                    byte_stream_s0 = torchac.encode_int16_normalized_cdf(
                        cdf_norm_s0, gt_s0.squeeze(-1).to(torch.int16).cpu()
                    )
                    byte_stream_ls.append(byte_stream_s0)
                    
                    # Stage 1 (conditioned on s0 results)
                    s1_emb = self.model.group2_pred_head_s1_emb(gt_s0.squeeze(-1))
                    group2_s1_F = enhanced_features + s1_emb
                    group2_s1_sparse = SparseTensor(coords=group2_coords, feats=group2_s1_F)
                    group2_s1_sparse = self.model.group2_spatial_conv_s1(group2_s1_sparse)
                    prob_s1 = self.model.group2_pred_head_s1(group2_s1_sparse.feats)
                    
                    cdf_s1 = torch.cat((prob_s1[:, 0:1]*0, prob_s1.cumsum(dim=-1)), dim=-1)
                    cdf_norm_s1 = op._convert_to_int_and_normalize(torch.clamp(cdf_s1, 0, 1), True).cpu()
                    byte_stream_s1 = torchac.encode_int16_normalized_cdf(
                        cdf_norm_s1, gt_s1.squeeze(-1).to(torch.int16).cpu()
                    )
                    byte_stream_ls.append(byte_stream_s1)
                else:
                    byte_stream_ls.extend([b''] * 2)
            
            byte_stream = op.pack_byte_stream_ls(byte_stream_ls)
            
            # Save base point cloud info
            base_x_coords, base_x_feats = data_ls[0]
            base_x_len = base_x_coords.shape[0]
            base_x_coords = base_x_coords[:, 1:].cpu().numpy()  # (n, 3)
            base_x_feats = base_x_feats.cpu().numpy()  # (n, 1)
            
            # Metadata encoded at the front: posQ(2) + use_tuned(1) + net_bits(8) + tuned_params_len(4) + tuned_params(...) + num_layers(1) + base_x_len(1) + base_x_coords + base_x_feats + byte_stream
            full_byte_stream = b''
            full_byte_stream += np.array(self.posQ, dtype=np.float16).tobytes()
            
            full_byte_stream += np.array(1 if use_finetuned_model else 0, dtype=np.uint8).tobytes()
            full_byte_stream += np.array(int(net_bits), dtype=np.uint64).tobytes() 
            full_byte_stream += np.array(len(tuned_params_bytes), dtype=np.uint32).tobytes() 
            full_byte_stream += tuned_params_bytes 
            
            full_byte_stream += np.array(num_layers, dtype=np.uint8).tobytes() 
            full_byte_stream += np.array(base_x_len, dtype=np.uint8).tobytes()
            full_byte_stream += np.array(base_x_coords, dtype=np.uint8).tobytes()
            full_byte_stream += np.array(base_x_feats, dtype=np.uint8).tobytes()
            full_byte_stream += byte_stream
            
            if self.device.startswith('cuda'):
                torch.cuda.synchronize() 
            enc_time_end = time.time()
            enc_time = enc_time_end - enc_time_start
            
            if return_metadata:
                metadata = {
                    'posQ': self.posQ,
                    'num_layers': num_layers,
                    'base_x_coords': base_x_coords,
                    'base_x_feats': base_x_feats,
                    'base_x_len': base_x_len,
                    'enc_time': enc_time,
                    'use_tuned': use_finetuned_model,
                    'net_bits': net_bits,
                    'tune_time': tune_time
                }
                return full_byte_stream, metadata
            else:
                return full_byte_stream, enc_time
    
    def decompress(self, byte_stream, lossy_level=None):
        """
        Decompress point cloud data.
        
        Args:
            byte_stream: Compressed byte stream (includes metadata and compressed data)
            lossy_level: Number of lossy levels. Uses self.lossy_level if None. 0 means lossless.
        
        Returns:
            xyz: numpy array, shape (N, 3) point cloud coordinates
            dec_time: float, decoding time (seconds)
        """
        with torch.no_grad():
            if self.device.startswith('cuda'):
                torch.cuda.synchronize()
            dec_time_start = time.time()

            # Parse metadata
            offset = 0
            posQ = np.frombuffer(byte_stream[offset:offset+2], dtype=np.float16)[0]
            offset += 2
            
            use_tuned_flag = np.frombuffer(byte_stream[offset:offset+1], dtype=np.uint8)[0]
            offset += 1
            
            net_bits = np.frombuffer(byte_stream[offset:offset+8], dtype=np.uint64)[0]
            offset += 8
            
            tuned_params_len = np.frombuffer(byte_stream[offset:offset+4], dtype=np.uint32)[0]
            offset += 4
            
            if use_tuned_flag > 0 and tuned_params_len > 0:
                tuned_params_bytes = byte_stream[offset:offset+tuned_params_len]
                offset += tuned_params_len
                
                try:
                    import deepCABAC
                    decoder = deepCABAC.Decoder()
                    decoder.getStream(np.frombuffer(tuned_params_bytes, dtype=np.uint8))
                    
                    state_dict = self.model.state_dict()
                    if len(self.postrain_ls) == 0:
                        postrain_ls = []
                        for group in [1, 2]:
                            for stage in range(2):
                                postrain_ls.append(f'group{group}_pred_head_s{stage}')
                    else:
                        postrain_ls = self.postrain_ls
                    
                    for full_param_name, param in self.model.named_parameters():
                        if any(full_param_name.startswith(module_name) for module_name in postrain_ls):
                            is_target_layer = '.2.' in full_param_name or '.4.' in full_param_name or '.0.' in full_param_name
                            if is_target_layer:
                                try:
                                    param_np = decoder.decodeWeights()
                                    state_dict[full_param_name] = torch.tensor(param_np).to(self.device)
                                except:
                                    pass 
                    
                    decoder.finish()
                    self.model.load_state_dict(state_dict, strict=False)
                    print(f"Loaded full fine-tuning parameters (Size: {net_bits/8:.2f} bytes)")
                except Exception as e:
                    print(f"Warning: Failed to load fine-tuning parameters: {e}, falling back to original model.")
            else:
                tuned_params_bytes = b''
            
            num_layers = np.frombuffer(byte_stream[offset:offset+4], dtype=np.uint8)[0]
            offset += 1
            
            base_x_len = np.frombuffer(byte_stream[offset:offset+4], dtype=np.uint8)[0]
            offset += 1
            
            base_x_coords = np.frombuffer(byte_stream[offset:offset+base_x_len*3], dtype=np.uint8)
            offset += base_x_len*3
            
            base_x_feats = np.frombuffer(byte_stream[offset:offset+base_x_len*1], dtype=np.uint8)
            offset += base_x_len*1
            
            compressed_byte_stream = byte_stream[offset:]
            
            base_x_coords = torch.tensor(base_x_coords.reshape(-1, 3), device=self.device, dtype=torch.int32)
            base_x_feats = torch.tensor(base_x_feats.reshape(-1, 1), device=self.device, dtype=torch.uint8)
            posQ = torch.tensor(posQ, device=self.device, dtype=torch.float32)
            
            # Initialize base point cloud
            x = SparseTensor(coords=torch.cat((base_x_feats.int()*0, base_x_coords), dim=-1), 
                           feats=base_x_feats.float()).to(self.device)
            
            current_lossy_level = lossy_level if lossy_level is not None else self.lossy_level
            byte_stream_ls = op.unpack_byte_stream(compressed_byte_stream)
            stream_idx = 0
            
            for depth in range(num_layers):
                if self.no_lossy_net and depth >= num_layers - current_lossy_level:
                    break
                
                is_lossy_layer = depth >= num_layers - current_lossy_level
                use_lossless_blocks = not is_lossy_layer
                net_for_blocks = self.model if use_lossless_blocks else self.lossy_model
                is_pure_lossy = is_lossy_layer
                
                if stream_idx >= len(byte_stream_ls):
                    break
                
                # Lossy decoding logic
                if is_pure_lossy:
                    if stream_idx >= len(byte_stream_ls):
                        break
                    k_bytes = byte_stream_ls[stream_idx]
                    stream_idx += 1
                    k = np.frombuffer(k_bytes, dtype=np.uint32)[0] if len(k_bytes) == 4 else 0
                    
                    if k > 0 and x.coords.shape[0] > 0 and self.lossy_model is not None:
                        # FCG required to fetch candidate points
                        
                        x_coords_sorted, x_feats_sorted = z_order_sort(x.coords, x.feats.unsqueeze(-1))
                        x_feats_sorted = x_feats_sorted.squeeze(-1)
                        x_O_dec = x_feats_sorted.int()
                        
                        if x_O_dec.dim() == 1:
                            x_O_dec = x_O_dec.unsqueeze(-1)
                        elif x_O_dec.dim() == 3:
                            x_O_dec = x_O_dec.squeeze(-1)
                            
                        x_F_dec = net_for_blocks.prior_embedding(x_O_dec).view(-1, net_for_blocks.channels)
                        x_dec_sparse = SparseTensor(coords=x_coords_sorted, feats=x_F_dec)
                        x_dec_sparse = net_for_blocks.prior_resnet(x_dec_sparse)
                        
                        x_up_C_dec, x_up_F_dec = net_for_blocks.fcg(x_coords_sorted, x_O_dec, x_F=x_dec_sparse.feats)
                        x_up_C_dec, x_up_F_dec = z_order_sort(x_up_C_dec, x_up_F_dec)
                        
                        x_up_F_dec = net_for_blocks.target_embedding(x_up_F_dec, x_up_C_dec)
                        x_up_dec = SparseTensor(coords=x_up_C_dec, feats=x_up_F_dec)
                        x_up_dec = net_for_blocks.target_resnet(x_up_dec)
                        
                        if x_up_dec.coords.shape[0] > 0:
                            if self.fixed_k:
                                # Fixed-k strategy via prob_children_groups and reverse transformation via fog
                                prob_parents = self.lossy_model.pred_head(x_up_dec.feats)  
                                
                                if not hasattr(self.lossy_model, 'occupancy_to_children'):
                                    o_to_c = torch.zeros(256, 8, device=self.device)
                                    for i in range(256):
                                        for j in range(8):
                                            if (i >> j) & 1:
                                                o_to_c[i, j] = 1
                                    self.lossy_model.register_buffer('occupancy_to_children', o_to_c, persistent=False)
                                
                                prob_children_groups = torch.matmul(prob_parents, self.lossy_model.occupancy_to_children)
                                num_potential_children = prob_children_groups.numel()
                                k_safe = min(int(k), num_potential_children)
                                
                                if k_safe > 0:
                                    _, top_indices_flat = torch.topk(prob_children_groups.view(-1), k_safe)
                                    parent_indices = torch.div(top_indices_flat, 8, rounding_mode='floor')
                                    child_indices = torch.remainder(top_indices_flat, 8)
                                    selected_parent_coords = x_up_dec.coords[parent_indices]
                                    
                                    coords_k_points = self.fcg_from_indices(selected_parent_coords, child_indices)
                                    
                                    # Key operation: Reverse transformation of generated points via fog
                                    temp_sparse = SparseTensor(
                                        coords=coords_k_points, 
                                        feats=torch.ones_like(coords_k_points[:, 0:1], dtype=torch.float)
                                    )
                                    x = self.model.fog(temp_sparse)
                                else:
                                    x = SparseTensor(
                                        coords=torch.empty(0, 4, dtype=torch.int32, device=self.device), 
                                        feats=torch.empty(0, 1, dtype=torch.int32, device=self.device)
                                    )
                            else:
                                # Non-fixed k strategy: top-k selection
                                prob = self.lossy_model.pred_head(x_up_dec.feats)  
                                prob_occupied = 1.0 - prob[:, 0]
                                k_safe = min(int(k), int(prob_occupied.shape[0]))
                                _, topk_indices = torch.topk(prob_occupied, k_safe)
                                topk_prob = prob[topk_indices]  
                                occ_hat = _top2_avoid_zero(topk_prob).to(torch.int16)  
                                
                                x_up_O_dec = torch.zeros(x_up_C_dec.shape[0], 1, device=self.device, dtype=torch.int16)
                                x_up_O_dec[topk_indices] = occ_hat.view(-1, 1)
                                
                                valid_mask = x_up_O_dec.squeeze(-1) > 0
                                next_coords = x_up_C_dec[valid_mask]
                                next_feats = x_up_O_dec[valid_mask].float().unsqueeze(-1)
                                next_coords, next_feats = z_order_sort(next_coords, next_feats.unsqueeze(-1))
                                next_feats = next_feats.squeeze(-1)
                                x = SparseTensor(coords=next_coords, feats=next_feats).to(self.device)
                        else:
                            x = SparseTensor(
                                coords=torch.empty(0, 4, dtype=torch.int32, device=self.device), 
                                feats=torch.empty(0, 1, dtype=torch.int32, device=self.device)
                            )
                    else:
                        x = SparseTensor(
                            coords=torch.empty(0, 4, dtype=torch.int32, device=self.device), 
                            feats=torch.empty(0, 1, dtype=torch.int32, device=self.device)
                        )
                    continue
                
                # Standard Lossless Decoding Flow
                x_coords_sorted, x_feats_sorted = z_order_sort(x.coords, x.feats.unsqueeze(-1))
                x_feats_sorted = x_feats_sorted.squeeze(-1)
                
                x_O = x_feats_sorted.int()
                if x_O.dim() == 1:
                    x_O = x_O.unsqueeze(-1)
                x_F = net_for_blocks.prior_embedding(x_O).view(-1, net_for_blocks.channels)
                x_dec_sparse = SparseTensor(coords=x_coords_sorted, feats=x_F)
                x_dec_sparse = net_for_blocks.prior_resnet(x_dec_sparse)
                
                x_up_C_dec, x_up_F_dec = net_for_blocks.fcg(x_coords_sorted, x_O, x_F=x_dec_sparse.feats)
                x_up_C_dec, x_up_F_dec = z_order_sort(x_up_C_dec, x_up_F_dec)
                
                x_up_F_dec = net_for_blocks.target_embedding(x_up_F_dec, x_up_C_dec)
                x_up_dec = SparseTensor(coords=x_up_C_dec, feats=x_up_F_dec)
                x_up_dec = net_for_blocks.target_resnet(x_up_dec)
                
                if not torch.equal(x_up_dec.coords, x_up_C_dec):
                    x_up_C_dec_out, x_up_F_dec_out = z_order_sort(x_up_dec.coords, x_up_dec.feats)
                    x_up_dec = SparseTensor(coords=x_up_C_dec_out, feats=x_up_F_dec_out)
                    x_up_C_dec = x_up_C_dec_out
                
                group1_mask_dec, group2_mask_dec = self.model.get_3d_checkerboard_groups(x_up_C_dec)
                x_up_O_dec = torch.zeros(x_up_C_dec.shape[0], dtype=torch.long, device=self.device)
                
                # 4 streams required in total (2 stages per group)
                total_streams = 4
                if stream_idx + total_streams > len(byte_stream_ls):
                    break
                
                byte_streams_g1 = byte_stream_ls[stream_idx:stream_idx+2]
                byte_streams_g2 = byte_stream_ls[stream_idx+2:stream_idx+4]
                stream_idx += 4
                
                # Group 1 Decoding
                if group1_mask_dec.sum() > 0:
                    group1_coords_dec = x_up_C_dec[group1_mask_dec]
                    group1_features_dec = x_up_dec.feats[group1_mask_dec]
                    
                    # Stage 0
                    group1_sparse = SparseTensor(coords=group1_coords_dec, feats=group1_features_dec)
                    group1_s0_sparse = self.model.group1_spatial_conv_s0(group1_sparse)
                    prob_s0 = self.model.group1_pred_head_s0(group1_s0_sparse.feats)
                    
                    if len(byte_streams_g1) > 0 and len(byte_streams_g1[0]) > 0:
                        cdf_s0 = torch.cat((prob_s0[:, 0:1]*0, prob_s0.cumsum(dim=-1)), dim=-1)
                        cdf_norm_s0 = op._convert_to_int_and_normalize(torch.clamp(cdf_s0, 0, 1), True).cpu()
                        decoded_s0 = torchac.decode_int16_normalized_cdf(cdf_norm_s0, byte_streams_g1[0]).to(self.device).long()
                    else:
                        decoded_s0 = torch.argmax(prob_s0, dim=-1)
                    
                    # Stage 1
                    s1_emb = self.model.group1_pred_head_s1_emb(decoded_s0)
                    group1_s1_F = group1_features_dec + s1_emb
                    group1_s1_sparse = SparseTensor(coords=group1_coords_dec, feats=group1_s1_F)
                    group1_s1_sparse = self.model.group1_spatial_conv_s1(group1_s1_sparse)
                    prob_s1 = self.model.group1_pred_head_s1(group1_s1_sparse.feats)
                    
                    if len(byte_streams_g1) > 1 and len(byte_streams_g1[1]) > 0:
                        cdf_s1 = torch.cat((prob_s1[:, 0:1]*0, prob_s1.cumsum(dim=-1)), dim=-1)
                        cdf_norm_s1 = op._convert_to_int_and_normalize(torch.clamp(cdf_s1, 0, 1), True).cpu()
                        decoded_s1 = torchac.decode_int16_normalized_cdf(cdf_norm_s1, byte_streams_g1[1]).to(self.device).long()
                    else:
                        decoded_s1 = torch.argmax(prob_s1, dim=-1)
                    
                    group1_occupancy_dec = decoded_s0 + decoded_s1 * 16
                    x_up_O_dec[group1_mask_dec] = group1_occupancy_dec
                
                # Group 2 Decoding
                if group2_mask_dec.sum() > 0:
                    if group1_mask_dec.sum() > 0:
                        enhanced_features_dec = self.model.aggregate_neighbor_features_efficient(
                            x_up_C_dec, x_up_dec.feats, group1_mask_dec, group2_mask_dec, x_up_O_dec
                        )
                        group2_coords_dec = x_up_C_dec[group2_mask_dec]
                    else:
                        group2_coords_dec = x_up_C_dec[group2_mask_dec]
                        enhanced_features_dec = x_up_dec.feats[group2_mask_dec]
                    
                    # Stage 0
                    group2_sparse = SparseTensor(coords=group2_coords_dec, feats=enhanced_features_dec)
                    group2_s0_sparse = self.model.group2_spatial_conv_s0(group2_sparse)
                    prob_s0 = self.model.group2_pred_head_s0(group2_s0_sparse.feats)
                    
                    if len(byte_streams_g2) > 0 and len(byte_streams_g2[0]) > 0:
                        cdf_s0 = torch.cat((prob_s0[:, 0:1]*0, prob_s0.cumsum(dim=-1)), dim=-1)
                        cdf_norm_s0 = op._convert_to_int_and_normalize(torch.clamp(cdf_s0, 0, 1), True).cpu()
                        decoded_s0 = torchac.decode_int16_normalized_cdf(cdf_norm_s0, byte_streams_g2[0]).to(self.device).long()
                    else:
                        decoded_s0 = torch.argmax(prob_s0, dim=-1)
                    
                    # Stage 1
                    s1_emb = self.model.group2_pred_head_s1_emb(decoded_s0)
                    group2_s1_F = enhanced_features_dec + s1_emb
                    group2_s1_sparse = SparseTensor(coords=group2_coords_dec, feats=group2_s1_F)
                    group2_s1_sparse = self.model.group2_spatial_conv_s1(group2_s1_sparse)
                    prob_s1 = self.model.group2_pred_head_s1(group2_s1_sparse.feats)
                    
                    if len(byte_streams_g2) > 1 and len(byte_streams_g2[1]) > 0:
                        cdf_s1 = torch.cat((prob_s1[:, 0:1]*0, prob_s1.cumsum(dim=-1)), dim=-1)
                        cdf_norm_s1 = op._convert_to_int_and_normalize(torch.clamp(cdf_s1, 0, 1), True).cpu()
                        decoded_s1 = torchac.decode_int16_normalized_cdf(cdf_norm_s1, byte_streams_g2[1]).to(self.device).long()
                    else:
                        decoded_s1 = torch.argmax(prob_s1, dim=-1)
                    
                    group2_occupancy_dec = decoded_s0 + decoded_s1 * 16
                    
                    # Map the sorted Group 2 results back to the global indices
                    group2_indices_in_full = torch.where(group2_mask_dec)[0]
                    sub_coords = x_up_C_dec[group2_mask_dec]
                    code = calc_morton_code(sub_coords)
                    sort_idx = torch.argsort(code)
                    sorted_indices_in_full = group2_indices_in_full[sort_idx]
                    x_up_O_dec[sorted_indices_in_full] = group2_occupancy_dec
                
                # Fetch next level points for decoding
                valid_mask_dec_full = x_up_O_dec > 0
                next_coords = x_up_C_dec[valid_mask_dec_full]
                next_feats = x_up_O_dec[valid_mask_dec_full].float().unsqueeze(-1)
                
                next_coords, next_feats = z_order_sort(next_coords, next_feats.unsqueeze(-1))
                next_feats = next_feats.squeeze(-1)
                x = SparseTensor(coords=next_coords, feats=next_feats).to(self.device)

            x_feats_for_fcg = x.feats.int()
            if x_feats_for_fcg.dim() == 3:
                x_feats_for_fcg = x_feats_for_fcg.squeeze(-1)
            scan = self.model.fcg(x.coords, x_feats_for_fcg, x_F=None)
            
            # Remove batch index column
            scan = scan[:, 1:]  
            
            # Dequantize
            scan = scan.float() * posQ
            if self.is_data_pre_quantized:
                scan = scan - self.preprocess_shift
            else:
                scan = (scan - self.preprocess_shift) * self.preprocess_scale
            
            if self.device.startswith('cuda'):
                torch.cuda.synchronize()
            dec_time_end = time.time()
            dec_time = dec_time_end - dec_time_start
            
            return scan.float().cpu().numpy(), dec_time
    
    def test(self, xyz):
        """
        Test encoding and decoding consistency.
        
        Args:
            xyz: numpy array or torch tensor, shape (N, 3) point cloud coordinates
        
        Returns:
            is_consistent: bool, whether reconstruction is exact
            original_xyz: numpy array, original point cloud
            decompressed_xyz: numpy array, decompressed point cloud
            stats: dict, statistics (points, compression ratio, etc.)
        """
        byte_stream, metadata = self.compress(xyz, return_metadata=True)
        enc_time = metadata['enc_time']
        
        decompressed_xyz, dec_time = self.decompress(byte_stream)
        
        if isinstance(xyz, torch.Tensor):
            original_xyz = xyz.cpu().numpy()
        else:
            original_xyz = np.array(xyz)
        
        # Quantize point cloud for comparison since compression involves quantization step
        original_scaled = original_xyz / self.preprocess_scale
        original_quantized = np.round((original_scaled + self.preprocess_shift) / metadata['posQ']).astype(np.int32)
        
        decompressed_scaled = decompressed_xyz / self.preprocess_scale
        decompressed_quantized = np.round((decompressed_scaled + self.preprocess_shift) / metadata['posQ']).astype(np.int32)
        
        # Point order may shift, sort to compare
        original_sorted = original_quantized[np.lexsort(original_quantized.T)]
        decompressed_sorted = decompressed_quantized[np.lexsort(decompressed_quantized.T)]
        
        num_points_match = original_sorted.shape[0] == decompressed_sorted.shape[0]
        
        if num_points_match:
            # Set comparison is robust to ordering
            original_set = set(tuple(row) for row in original_sorted)
            decompressed_set = set(tuple(row) for row in decompressed_sorted)
            is_consistent = original_set == decompressed_set
            
            # Need strict length counter if duplicates might exist
            if is_consistent and len(original_set) != original_sorted.shape[0]:
                original_counter = Counter(tuple(row) for row in original_sorted)
                decompressed_counter = Counter(tuple(row) for row in decompressed_sorted)
                is_consistent = original_counter == decompressed_counter
        else:
            is_consistent = False
        
        original_size = original_xyz.nbytes if isinstance(original_xyz, np.ndarray) else original_xyz.numel() * 4
        compressed_size = len(byte_stream)
        
        stats = {
            'original_num_points': original_sorted.shape[0],
            'decompressed_num_points': decompressed_sorted.shape[0],
            'num_points_match': num_points_match,
            'original_size_bytes': original_size,
            'compressed_size_bytes': compressed_size,
            'bpp': compressed_size * 8 / original_sorted.shape[0] if original_sorted.shape[0] > 0 else 0,
            'enc_time': enc_time,
            'dec_time': dec_time,
            'tune_time': metadata.get('tune_time', -1.0),
            'net_bits': metadata.get('net_bits', 0),
            'use_tuned': metadata.get('use_tuned', False),
        }
        
        return is_consistent, original_xyz, decompressed_xyz, stats