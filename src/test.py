import os
import time
import random
import argparse
import tempfile
import re

import numpy as np
from glob import glob
from tqdm import tqdm

import torch
import torch.nn as nn

from torchsparse.nn import functional as F

import kit.io as io
import kit.op as op
import pandas as pd

from coder.coder_intra import AnyPcc_CoderIntra

try:
    from pc_error import pc_error
except ImportError:
    print("Warning: 'pc_error' module not found. PSNR calculation will be skipped.")
    pc_error = None

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
device = 'cuda'

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set torchsparse config
conv_config = F.conv_config.get_default_conv_config()
conv_config.kmap_mode = "hashmap"
F.conv_config.set_global_conv_config(conv_config)

parser = argparse.ArgumentParser(
    prog='test.py',
    description='Test point cloud codec using AnyPcc_CoderIntra (supports lossless/lossy)',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--input_glob', nargs='+', required=True, help='Path(s) to one or more input point cloud folders.')
parser.add_argument("--is_data_pre_quantized", type=bool, default=False, help="Whether the input data is pre-quantized.")
parser.add_argument('--posQ', default=1, type=int, help='Quantization scale.')
parser.add_argument('--preprocess_scale', type=float, default=1.0, help='Preprocess scale.')
parser.add_argument('--preprocess_shift', type=float, default=0.0, help='Preprocess shift.')

parser.add_argument('--channels', type=int, default=32, help='Neural network channels.')
parser.add_argument('--kernel_size', type=int, default=3, help='Convolution kernel size.')
parser.add_argument('--ckpt', required=True, help='Path to lossless model checkpoint.')
parser.add_argument('--res', type=int, default=1024, help='Voxel resolution for PSNR calculation (default 1024).')

# Lossy parameters
parser.add_argument('--lossy_ckpt', type=str, default=None, help='Path to lossy model checkpoint (required if lossy_level > 0, unless --no_lossy_net is used).')
parser.add_argument('--lossy_level', type=int, default=0, help='Lossy level: 0=pure lossless, >0=lossy (number indicates the number of truncated levels).')
parser.add_argument('--no_lossy_net', action='store_true', help='Pure truncation mode (do not use lossy network, only truncate levels).')
parser.add_argument('--fixed_k', action='store_true', help='Fixed k strategy: decode a fixed k number of points.')

# Fine-tuning parameters
parser.add_argument('--tune', action='store_true', help='Enable full fine-tuning (sample adaptive).')
parser.add_argument('--postrain_lr', type=float, default=0.004, help='Learning rate for full fine-tuning.')
parser.add_argument('--postrain_ls', nargs='*', default=[], help='List of module names to fine-tune (default auto-detects all classification heads).')
parser.add_argument('--epoch', type=int, default=100, help='Number of epochs for full fine-tuning.')
parser.add_argument('--tune_threshold', type=int, default=20000, help='Minimum number of points to trigger fine-tuning.')

# Testing & Output parameters
parser.add_argument('--num_samples', default=-1, type=int, help='Randomly select samples for quick testing. [-1 for all data].')
parser.add_argument('--resultdir', type=str, default='./results_test', help='Directory to save result CSV files.')
parser.add_argument('--prefix', type=str, default='test', help='Prefix for the result CSV files.')
parser.add_argument('--compress_only', action='store_true', help='Only perform compression, skip decompression and consistency check (valid in lossless mode).')
parser.add_argument('--save_bin', type=str, default=None, help='Directory to save encoded bin files.')
parser.add_argument('--save_decoded', type=str, default=None, help='Directory to save reconstructed point clouds.')

args = parser.parse_args()

# Parameter validation
if args.lossy_level > 0:
    if args.lossy_ckpt is None and not args.no_lossy_net:
        raise ValueError(f"Lossy mode (lossy_level={args.lossy_level}) requires --lossy_ckpt parameter, or enable --no_lossy_net for pure truncation mode.")
    if args.lossy_ckpt is not None and not os.path.exists(args.lossy_ckpt):
        raise FileNotFoundError(f"Lossy model checkpoint does not exist: {args.lossy_ckpt}")
elif args.lossy_level == 0:
    if args.lossy_ckpt is not None:
        print("Warning: In lossless mode (lossy_level=0), --lossy_ckpt will be ignored.")

if args.compress_only and args.lossy_level > 0:
    raise ValueError("--compress_only mode is only valid in lossless mode (lossy_level=0).")
if args.compress_only:
    print("Info: Compress-only mode enabled. Decompression and consistency check will be skipped.")

os.makedirs(args.resultdir, exist_ok=True)

if args.save_bin is not None:
    os.makedirs(args.save_bin, exist_ok=True)
    print(f"Info: Encoded bin files will be saved to: {args.save_bin}")

# Load file paths
file_path_ls = []
for input_path in args.input_glob:
    files_in_path = glob(os.path.join(input_path, '**', '*.*'), recursive=True)
    file_path_ls.extend(files_in_path)

file_path_ls = sorted(list(set(file_path_ls)))
file_path_ls = [f for f in file_path_ls if f.endswith('h5') or f.endswith('.ply') or f.endswith('bin') or f.endswith('npy')]

if args.num_samples > 0:
    file_path_ls = file_path_ls[:args.num_samples]

xyz_ls = io.read_point_clouds(file_path_ls)

coder = AnyPcc_CoderIntra(
    model_path=args.ckpt,
    device=device,
    channels=args.channels,
    kernel_size=args.kernel_size,
    posQ=args.posQ,
    preprocess_scale=args.preprocess_scale,
    preprocess_shift=args.preprocess_shift,
    is_data_pre_quantized=args.is_data_pre_quantized,
    lossy_model_path=args.lossy_ckpt,
    lossy_level=args.lossy_level,
    no_lossy_net=args.no_lossy_net,
    fixed_k=args.fixed_k,
    tune=args.tune,
    postrain_lr=args.postrain_lr,
    postrain_ls=args.postrain_ls if len(args.postrain_ls) > 0 else None,
    epoch=args.epoch,
    tune_threshold=args.tune_threshold,
)

all_results = None
all_results_with_avg = None

with torch.no_grad():
    for file_idx in tqdm(range(len(file_path_ls))):
        file_path = file_path_ls[file_idx]
        file_name = os.path.split(file_path)[-1]

        # Dynamically adjust resolution based on filename
        res = args.res
        match = re.search(r"vox(\d+)", file_name)
        if match:
            vox_level = int(match.group(1))
            res = int(1024 * (2 ** (vox_level - 10)))
        
        xyz = xyz_ls[file_idx]
        
        if args.compress_only:
            byte_stream, metadata = coder.compress(xyz, return_metadata=True)
            enc_time = metadata['enc_time']
            
            if args.save_bin is not None:
                bin_filename = os.path.splitext(file_name)[0] + '.bin'
                bin_filepath = os.path.join(args.save_bin, bin_filename)
                with open(bin_filepath, 'wb') as f:
                    f.write(byte_stream)
            
            if isinstance(xyz, torch.Tensor):
                original_xyz = xyz.cpu().numpy()
            else:
                original_xyz = np.array(xyz)
            
            original_size = original_xyz.nbytes if isinstance(original_xyz, np.ndarray) else original_xyz.numel() * 4
            compressed_size = len(byte_stream)
            num_points = original_xyz.shape[0] if len(original_xyz.shape) > 0 else 0
            bpp = compressed_size * 8 / num_points if num_points > 0 else 0
            
            stats = {
                'original_num_points': num_points,
                'decompressed_num_points': num_points, 
                'num_points_match': True, 
                'bpp': bpp,
                'enc_time': enc_time,
                'dec_time': 0.0, 
                'original_size_bytes': original_size,
                'compressed_size_bytes': compressed_size,
                'tune_time': metadata.get('tune_time', -1.0),
                'net_bits': metadata.get('net_bits', 0),
            }
            
            tune_time = stats.get('tune_time', -1.0)
            net_bits = stats.get('net_bits', 0)
            net_bpp = (net_bits / stats['original_num_points']) if stats['original_num_points'] > 0 and net_bits > 0 else -1.0
            
            is_consistent = True
            decompressed_xyz = None
            d1_psnr, d2_psnr = -1.0, -1.0
        else:
            if args.save_bin is not None:
                byte_stream, _ = coder.compress(xyz, return_metadata=False)
                bin_filename = os.path.splitext(file_name)[0] + '.bin'
                bin_filepath = os.path.join(args.save_bin, bin_filename)
                with open(bin_filepath, 'wb') as f:
                    f.write(byte_stream)
            
            is_consistent, original_xyz, decompressed_xyz, stats = coder.test(xyz)
            
            if args.lossy_level > 0:
                is_consistent = True
            
            tune_time = stats.get('tune_time', -1.0)
            net_bits = stats.get('net_bits', 0)
            net_bpp = (net_bits / stats['original_num_points']) if stats['original_num_points'] > 0 and net_bits > 0 else -1.0

            d1_psnr, d2_psnr = -1.0, -1.0
            if pc_error is not None and args.lossy_level > 0 and decompressed_xyz is not None and decompressed_xyz.shape[0] > 0:
                with tempfile.NamedTemporaryFile(suffix=".ply", delete=True) as tmp_file:
                    io.save_ply_ascii_geo(decompressed_xyz.astype(np.float32), tmp_file.name)
                    try:
                        metrics = pc_error(file_path, tmp_file.name, res=res, normal=False, show=False)
                        d1_psnr = metrics.get("mseF,PSNR (p2point)", [-1])[0]
                        d2_psnr = metrics.get("mseF,PSNR (p2plane)", [-1])[0]
                    except Exception:
                        pass

            if args.save_decoded is not None and decompressed_xyz is not None and decompressed_xyz.shape[0] > 0:
                decoded_filename = os.path.splitext(file_name)[0] + '_decoded.ply'
                decoded_path = os.path.join(args.save_decoded, decoded_filename)
                io.save_ply_ascii_geo(decompressed_xyz.astype(np.float32), decoded_path)
        
        results = {
            'filedir': file_name,
            'is_consistent': is_consistent,
            'original_num_points': stats['original_num_points'],
            'decompressed_num_points': stats['decompressed_num_points'],
            'num_points_match': stats['num_points_match'],
            'bpp': stats['bpp'],
            'enc_time': stats['enc_time'],
            'dec_time': stats['dec_time'],
            'original_size_bytes': stats['original_size_bytes'],
            'compressed_size_bytes': stats['compressed_size_bytes'],
            'd1_psnr': d1_psnr,
            'd2_psnr': d2_psnr,
            'lossy_level': args.lossy_level,
            'no_lossy_net': int(args.no_lossy_net),
            'has_lossy_model': int(args.lossy_ckpt is not None),
            'tune': int(args.tune),
            'tune_threshold': args.tune_threshold if args.tune else -1,
            'tune_time': tune_time,
            'net_bpp': net_bpp,
            'compress_only': int(args.compress_only),
        }
        
        results_df = pd.DataFrame([results])
        
        if file_idx == 0:
            all_results = results_df.copy(deep=True)
        else:
            all_results = pd.concat([all_results, results_df], ignore_index=True)
        
        if not all_results.empty:
            average_results = all_results.mean(numeric_only=True).to_dict()
            average_results['filedir'] = 'avg'
            if args.lossy_level == 0:
                average_results['is_consistent'] = all_results['is_consistent'].all()
            else:
                average_results['is_consistent'] = True 
            average_results['num_points_match'] = all_results['num_points_match'].all()
            average_results['lossy_level'] = args.lossy_level
            average_results['no_lossy_net'] = int(args.no_lossy_net)
            average_results['has_lossy_model'] = int(args.lossy_ckpt is not None)
            average_results['tune'] = int(args.tune)
            average_results['tune_threshold'] = args.tune_threshold if args.tune else -1
            average_results['compress_only'] = int(args.compress_only)
            
            average_df = pd.DataFrame([average_results])
            all_results_with_avg = pd.concat([all_results, average_df], ignore_index=True)
        else:
            all_results_with_avg = all_results
        
        csvfile = os.path.join(args.resultdir, args.prefix + '_data' + str(len(file_path_ls)) + '.csv')
        all_results_with_avg.to_csv(csvfile, index=False)
        
        status = "✓" if is_consistent else "✗"
        
        mode_parts = []
        if args.compress_only:
            mode_parts.append("Compress-only")
        if args.lossy_level == 0:
            mode_parts.append("Lossless")
        elif args.no_lossy_net:
            mode_parts.append(f"Lossy-Truncation (lossy_level={args.lossy_level})")
        else:
            mode_parts.append(f"Lossy-Standard (lossy_level={args.lossy_level})")
        
        if args.tune:
            if stats['original_num_points'] >= args.tune_threshold:
                mode_parts.append("Fine-tuned")
            else:
                mode_parts.append(f"No-tune (points < {args.tune_threshold})")
        
        mode_str = ", ".join(mode_parts)

        if args.lossy_level > 0 and d1_psnr >= 0:
            psnr_str = f", D1_PSNR={d1_psnr:.4f}, D2_PSNR={d2_psnr:.4f}"
        else:
            psnr_str = ""

        tune_str = ""
        if args.tune and stats.get('tune_time', -1) >= 0:
            tune_str = f", Tune time={stats['tune_time']:.2f}s"
            if stats.get('net_bpp', -1) > 0:
                tune_str += f", Model BPP={stats['net_bpp']:.4f}"
        
        if args.compress_only:
            tqdm.write(
                f"[{status}] {file_name} ({mode_str}): "
                f"Points={stats['original_num_points']}, "
                f"BPP={stats['bpp']:.4f}, "
                f"Enc time={stats['enc_time']:.3f}s"
                f"{tune_str}"
            )
        else:
            consistency_str = f", Consistent={is_consistent}" if args.lossy_level == 0 and args.posQ==1 else ""
            tqdm.write(
                f"[{status}] {file_name} ({mode_str}){consistency_str}: "
                f"Points={stats['original_num_points']}, "
                f"BPP={stats['bpp']:.4f}, "
                f"Enc time={stats['enc_time']:.3f}s, "
                f"Dec time={stats['dec_time']:.3f}s"
                f"{tune_str}"
                f"{psnr_str}"
            )

# Final save and print
if all_results is not None and not all_results.empty:
    if all_results_with_avg is not None and not all_results_with_avg.empty:
        final_results = all_results_with_avg
    else:
        average_results = all_results.mean(numeric_only=True).to_dict()
        average_results['filedir'] = 'avg'
        if args.lossy_level == 0:
            average_results['is_consistent'] = all_results['is_consistent'].all()
        else:
            average_results['is_consistent'] = True 
        average_results['num_points_match'] = all_results['num_points_match'].all()
        average_results['lossy_level'] = args.lossy_level
        average_results['no_lossy_net'] = int(args.no_lossy_net)
        average_results['has_lossy_model'] = int(args.lossy_ckpt is not None)
        average_results['tune'] = int(args.tune)
        average_results['tune_threshold'] = args.tune_threshold if args.tune else -1
        average_results['compress_only'] = int(args.compress_only)
        
        average_df = pd.DataFrame([average_results])
        final_results = pd.concat([all_results, average_df], ignore_index=True)
    
    csvfile = os.path.join(args.resultdir, args.prefix + '_data' + str(len(file_path_ls)) + '.csv')
    final_results.to_csv(csvfile, index=False)
    
    all_results_for_stats = final_results.iloc[:-1] if len(final_results) > 0 else final_results
    average_results = final_results.iloc[-1].to_dict() if len(final_results) > 0 else {}
    
    print('\n' + '='*80)
    print('Test Summary:')
    print('='*80)
    if args.compress_only:
        print(f'Test Mode: Compress-only (Quick BPP Test)')
    print(f'Test Mode: {"Lossless" if args.lossy_level == 0 else "Lossy"}')
    if args.lossy_level > 0:
        print(f'  Lossy levels: {args.lossy_level}')
        if args.no_lossy_net:
            print(f'  Pure truncation mode: Yes')
    if args.tune:
        print(f'Fine-tuning: Enabled')
        print(f'  Tune threshold: {args.tune_threshold} points')
    if args.save_decoded is not None:
        print(f'Save decoded point clouds: Yes (Path: {args.save_decoded})')
        print(f'  LR: {args.postrain_lr}, Epochs: {args.epoch}')
    if args.save_bin is not None:
        print(f'Save bin files: Yes (Path: {args.save_bin})')
    print(f'Total files: {len(file_path_ls)}')
    if not args.compress_only and args.lossy_level == 0:
        is_consistent_sum = all_results_for_stats['is_consistent'].sum() if len(all_results_for_stats) > 0 else 0
        print(f'Consistency checks passed: {is_consistent_sum}/{len(file_path_ls)}')
    bpp_val = average_results['bpp']
    enc_time_val = average_results['enc_time']
    dec_time_val = average_results['dec_time']
    print(f'Average BPP: {bpp_val:.4f}')
    if args.tune:
        tune_time_avg = average_results.get('tune_time', -1)
        net_bpp_avg = average_results.get('net_bpp', -1)
        if tune_time_avg >= 0:
            print(f'Average tune time: {tune_time_avg:.3f}s')
        if net_bpp_avg > 0:
            print(f'Average model BPP: {net_bpp_avg:.4f}')
    print(f'Average enc time: {enc_time_val:.3f}s')
    if not args.compress_only:
        print(f'Average dec time: {dec_time_val:.3f}s')
    print(f'Max GPU memory: {torch.cuda.max_memory_allocated()/1024/1024:.2f}MB')
    print(f'Results saved to: {csvfile}')
    print('='*80)
    
    if not args.compress_only:
        if args.lossy_level == 0:
            if len(all_results_for_stats) > 0 and not all_results_for_stats['is_consistent'].all():
                print('\nWarning: Some files failed the consistency check!')
                inconsistent_files = all_results_for_stats[~all_results_for_stats['is_consistent']]['filedir'].tolist()
                print(f'Inconsistent files: {inconsistent_files}')
            elif len(all_results_for_stats) > 0:
                print('\n✓ All files passed consistency check!')
        else:
            print('\nInfo: Consistency check skipped in lossy mode.')
    else:
        print('\nInfo: Compress-only mode enabled, consistency check skipped.')
else:
    print('Error: No files processed!')