"""
Read the bin files saved by compress.py and decode them into point clouds (saved as PLY).
Decoder parameters must match those used during compression (same ckpt, lossy configuration, etc.).
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
    prog="decompress.py",
    description="Read bin files saved by compress.py and decode them into point clouds",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument("--input_glob", nargs="+", required=True, help="One or more bin files or directories (will recursively search for .bin)")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the decoded point cloud PLY files")
parser.add_argument("--posQ", default=1, type=int, help="Quantization scale (must match compression; some info is parsed from the bin header)")
parser.add_argument("--preprocess_scale", type=float, default=1.0, help="Preprocess scale (must match compression)")
parser.add_argument("--preprocess_shift", type=float, default=0.0, help="Preprocess shift (must match compression)")

parser.add_argument("--channels", type=int, default=32, help="Number of neural network channels (must match compression)")
parser.add_argument("--kernel_size", type=int, default=3, help="Convolution kernel size (must match compression)")
parser.add_argument("--ckpt", required=True, help="Lossless model checkpoint path (must match compression)")

parser.add_argument("--lossy_ckpt", type=str, default=None, help="Lossy model checkpoint (required if the bin is lossy compressed)")
parser.add_argument("--lossy_level", type=int, default=0, help="Lossy level (must match compression)")
parser.add_argument("--no_lossy_net", action="store_true", help="Pure truncation mode (must match compression)")
parser.add_argument("--fixed_k", action="store_true", help="Fixed k strategy (must match compression)")

parser.add_argument("--num_samples", type=int, default=-1, help="Only decode the first N files, -1 for all")
parser.add_argument("--suffix", type=str, default="_decoded.ply", help="Output PLY filename suffix")

args = parser.parse_args()

if args.lossy_level > 0:
    if args.lossy_ckpt is None and not args.no_lossy_net:
        raise ValueError(
            f"Lossy bin decoding (lossy_level={args.lossy_level}) requires --lossy_ckpt or --no_lossy_net"
        )
    if args.lossy_ckpt is not None and not os.path.exists(args.lossy_ckpt):
        raise FileNotFoundError(f"Lossy model does not exist: {args.lossy_ckpt}")

os.makedirs(args.output_dir, exist_ok=True)

# Collect bin files
bin_path_ls = []
for input_path in args.input_glob:
    if os.path.isfile(input_path):
        if input_path.endswith(".bin"):
            bin_path_ls.append(input_path)
    else:
        found = glob(os.path.join(input_path, "**", "*.bin"), recursive=True)
        bin_path_ls.extend(found)

bin_path_ls = sorted(list(set(bin_path_ls)))

if args.num_samples > 0:
    bin_path_ls = bin_path_ls[: args.num_samples]

if not bin_path_ls:
    raise SystemExit("Error: No .bin files found")

# Decoder (do not enable tune, decoding only)
coder = AnyPcc_CoderIntra(
    model_path=args.ckpt,
    device=device,
    channels=args.channels,
    kernel_size=args.kernel_size,
    posQ=args.posQ,
    preprocess_scale=args.preprocess_scale,
    preprocess_shift=args.preprocess_shift,
    is_data_pre_quantized=False,
    lossy_model_path=args.lossy_ckpt,
    lossy_level=args.lossy_level,
    no_lossy_net=args.no_lossy_net,
    fixed_k=args.fixed_k,
    tune=False,
)

with torch.no_grad():
    for bin_path in tqdm(bin_path_ls, desc="Decoding"):
        with open(bin_path, "rb") as f:
            byte_stream = f.read()

        xyz, dec_time = coder.decompress(byte_stream)

        base_name = os.path.splitext(os.path.basename(bin_path))[0]
        out_name = base_name + (args.suffix if args.suffix.endswith(".ply") else args.suffix + ".ply")
        out_path = os.path.join(args.output_dir, out_name)

        xyz = np.asarray(xyz, dtype=np.float32)
        io.save_ply_ascii_geo(xyz, out_path)

        n_pts = xyz.shape[0]
        tqdm.write(f"{os.path.basename(bin_path)} -> {out_name} | Points={n_pts}, Decoding time={dec_time:.3f}s")

print(f"\nDecoding complete. Total {len(bin_path_ls)} files. PLY files saved to: {args.output_dir}")