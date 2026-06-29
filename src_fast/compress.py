"""
Compress point clouds and save them as bin files.
Uses the same encoder as test.py to compress specified point clouds and write them to the --output_dir.
"""
import os
import random
import argparse
from glob import glob

import numpy as np
from tqdm import tqdm
import torch
from torchsparse.nn import functional as F

import kit.io as io
from coder.coder_intra import AnyPcc_CoderIntra

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
device = "cuda"
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

conv_config = F.conv_config.get_default_conv_config()
conv_config.kmap_mode = "hashmap"
F.conv_config.set_global_conv_config(conv_config)

parser = argparse.ArgumentParser(
    prog="compress.py",
    description="Compress point clouds and save as bin files (to be decoded by decompress.py)",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument("--input_glob", nargs="+", required=True, help="One or more input point cloud paths (files or directories)")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the compressed bin files")
parser.add_argument("--is_data_pre_quantized", type=bool, default=False, help="Whether the input data is pre-quantized")
parser.add_argument("--posQ", default=1, type=int, help="Quantization scale")
parser.add_argument("--preprocess_scale", type=float, default=1.0, help="Preprocess scale")
parser.add_argument("--preprocess_shift", type=float, default=0.0, help="Preprocess shift")

parser.add_argument("--channels", type=int, default=32, help="Number of neural network channels")
parser.add_argument("--kernel_size", type=int, default=3, help="Convolution kernel size")
parser.add_argument("--ckpt", required=True, help="Lossless model checkpoint path")

parser.add_argument("--lossy_ckpt", type=str, default=None, help="Lossy model checkpoint (used when lossy_level > 0)")
parser.add_argument("--lossy_level", type=int, default=0, help="Lossy level: 0=lossless, >0=lossy")
parser.add_argument("--no_lossy_net", action="store_true", help="Pure truncation mode, without using the lossy network")
parser.add_argument("--fixed_k", action="store_true", help="Fixed k strategy")

parser.add_argument("--tune", action="store_true", help="Enable full fine-tuning")
parser.add_argument("--postrain_lr", type=float, default=0.004, help="Learning rate for full fine-tuning")
parser.add_argument("--postrain_ls", nargs="*", default=[], help="List of module names to fine-tune")
parser.add_argument("--epoch", type=int, default=100, help="Number of epochs for full fine-tuning")
parser.add_argument("--tune_threshold", type=int, default=20000, help="Minimum number of points to trigger fine-tuning")

parser.add_argument("--num_samples", type=int, default=-1, help="Only process the first N files, -1 for all")
parser.add_argument("--suffix", type=str, default=".bin", help="Output filename suffix (default .bin)")

args = parser.parse_args()

if args.lossy_level > 0:
    if args.lossy_ckpt is None and not args.no_lossy_net:
        raise ValueError(
            f"Lossy mode (lossy_level={args.lossy_level}) requires --lossy_ckpt or --no_lossy_net"
        )
    if args.lossy_ckpt is not None and not os.path.exists(args.lossy_ckpt):
        raise FileNotFoundError(f"Lossy model does not exist: {args.lossy_ckpt}")

os.makedirs(args.output_dir, exist_ok=True)

# Collect input files
file_path_ls = []
for input_path in args.input_glob:
    if os.path.isfile(input_path):
        file_path_ls.append(input_path)
    else:
        files_in_path = glob(os.path.join(input_path, "**", "*.*"), recursive=True)
        file_path_ls.extend(files_in_path)

file_path_ls = sorted(list(set(file_path_ls)))
file_path_ls = [
    f for f in file_path_ls
    if f.endswith(".h5") or f.endswith(".ply") or f.endswith(".bin") or f.endswith(".npy")
]

if args.num_samples > 0:
    file_path_ls = file_path_ls[: args.num_samples]

if not file_path_ls:
    raise SystemExit("Error: No point cloud files found")

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
    postrain_ls=args.postrain_ls if args.postrain_ls else None,
    epoch=args.epoch,
    tune_threshold=args.tune_threshold,
)

with torch.no_grad():
    for file_idx in tqdm(range(len(file_path_ls)), desc="Compressing"):
        file_path = file_path_ls[file_idx]
        file_name = os.path.basename(file_path)
        base_name = os.path.splitext(file_name)[0]
        xyz = xyz_ls[file_idx]

        byte_stream, metadata = coder.compress(xyz, return_metadata=True)

        out_name = base_name + (args.suffix if args.suffix.startswith(".") else "." + args.suffix)
        out_path = os.path.join(args.output_dir, out_name)
        with open(out_path, "wb") as f:
            f.write(byte_stream)

        n_pts = xyz.shape[0] if hasattr(xyz, "shape") else len(xyz)
        bpp = len(byte_stream) * 8 / n_pts if n_pts > 0 else 0
        tqdm.write(
            f"{file_name} -> {out_name} | Points={n_pts}, BPP={bpp:.4f}, Enc time={metadata.get('enc_time', 0):.3f}s"
        )

print(f"\nCompression complete. Total {len(file_path_ls)} files. Bin files saved to: {args.output_dir}")