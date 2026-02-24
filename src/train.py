import os
import random
import argparse
import datetime
import logging
from logging.handlers import RotatingFileHandler
import json

import numpy as np
from glob import glob

import torch
import torch.utils.data
from torch.utils.data import ConcatDataset
from torch import nn
from torch.cuda import amp

from torchsparse.nn import functional as F
from torchsparse.utils.collate import sparse_collate_fn

# Optional WandB import
try:
    import wandb
except ImportError:
    wandb = None

import swanlab
swanlab.sync_wandb()

from dataset import UnifiedPCDataset

# Reproducibility
seed = 11
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = 'cuda:0'

# Set torchsparse config
conv_config = F.conv_config.get_default_conv_config()
conv_config.kmap_mode = "hashmap"
F.conv_config.set_global_conv_config(conv_config)

def custom_collate_fn(batch):
    """
    Custom collate function to preserve dataset_name and dataset_idx info.
    """
    collated = sparse_collate_fn(batch)
    
    if 'dataset_name' in batch[0]:
        dataset_names = [item.get('dataset_name', 'unknown') for item in batch]
        dataset_indices = [item.get('dataset_idx', -1) for item in batch]
        collated['dataset_name'] = dataset_names[0] if len(set(dataset_names)) == 1 else dataset_names
        collated['dataset_idx'] = dataset_indices[0] if len(set(dataset_indices)) == 1 else dataset_indices
    
    return collated

def get_files_from_config(ds_config, is_training=True):
    """
    Get, filter, and split file lists based on JSON configuration.
    """
    all_files = []
    for data_dir in ds_config['data_dirs']:
        if not os.path.exists(data_dir):
            logger.warning(f"Data directory does not exist: {data_dir}")
            continue
        search_pattern = os.path.join(data_dir, ds_config['file_pattern'])
        dir_files = glob(search_pattern, recursive=True)
        all_files.extend(dir_files)
    
    if not all_files:
        return []

    all_files.sort() # Ensure consistent order

    split_mode = ds_config.get('split_mode')
    split_range = ds_config.get('split_range')
    
    if split_mode == 'range' and split_range:
        start_idx, end_idx = split_range
        if end_idx == -1: end_idx = len(all_files)
        selected_files = all_files[start_idx:end_idx]
        logger.info(f"Dataset '{ds_config['name']}' range split [{start_idx}:{end_idx}]: {len(selected_files)} files.")
        all_files = selected_files

    elif split_mode == 'sequence_split' and split_range:
        start_seq, end_seq = split_range
        # Assuming sequence ID is a zero-padded 2-digit number in the path
        sequence_ids = {f"{i:02d}" for i in range(start_seq, end_seq + 1)}
        
        selected_files = []
        for file_path in all_files:
            if any(part in sequence_ids for part in file_path.split(os.sep)):
                selected_files.append(file_path)
        
        logger.info(f"Dataset '{ds_config['name']}' sequence split [{start_seq}-{end_seq}]: {len(selected_files)} files.")
        all_files = selected_files
        
    max_samples = ds_config.get('max_samples')
    if max_samples and len(all_files) > max_samples:
        if is_training:
            random.shuffle(all_files)
            logger.info(f"Dataset '{ds_config['name']}' random sampled {max_samples} files.")
        else:
            logger.info(f"Dataset '{ds_config['name']}' took first {max_samples} files.")
        all_files = all_files[:max_samples]
        
    return all_files


class WeightedMultiDataset(torch.utils.data.Dataset):
    """
    Dataset class to handle multiple datasets from JSON config with weighted sampling.
    """
    def __init__(self, config_path, **global_dataset_kwargs):
        super().__init__()
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.sub_datasets = []
        self.weights = []
        self.dataset_names = []
        self.dataset_indices = []
        self.total_length = 0

        logger.info(f"Loading datasets from '{config_path}'...")
        for idx, ds_config in enumerate(self.config['datasets']):
            all_files = get_files_from_config(ds_config, is_training=True)
            
            if not all_files:
                continue

            current_kwargs = global_dataset_kwargs.copy()
            overrides = {}
            for key, value in ds_config.items():
                if key in current_kwargs:
                    current_kwargs[key] = value
                    overrides[key] = value
            
            if overrides:
                logger.info(f" - Dataset '{ds_config['name']}' specific config: {overrides}")

            sub_dataset = UnifiedPCDataset(all_files, **current_kwargs)
            self.sub_datasets.append(sub_dataset)
            self.dataset_names.append(ds_config['name'])
            self.dataset_indices.append(len(self.sub_datasets) - 1)

            # Weight determines sampling probability
            weight = ds_config.get('sampling_weight', 1.0)
            self.weights.append(weight)
            self.total_length += len(sub_dataset)
            
            logger.info(f" - Loaded '{ds_config['name']}' (weight: {weight}), count: {len(sub_dataset)}")

        if not self.sub_datasets:
            raise ValueError("Error: No datasets loaded.")

        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]

    def __len__(self):
        return self.total_length

    def __getitem__(self, index):
        # Sample dataset based on weights
        chosen_idx = random.choices(range(len(self.sub_datasets)), weights=self.weights, k=1)[0]
        chosen_dataset = self.sub_datasets[chosen_idx]
        dataset_name = self.dataset_names[chosen_idx]
        
        # Sample random file from chosen dataset
        random_idx = random.randint(0, len(chosen_dataset) - 1)
        sample = chosen_dataset[random_idx]
        
        sample['dataset_name'] = dataset_name
        sample['dataset_idx'] = chosen_idx
        
        return sample

# ======================== Argument Parsing ========================

parser = argparse.ArgumentParser(
    prog='train.py',
    description='Training from scratch.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--train_config', type=str, required=True, help='Path to training JSON config.')
parser.add_argument('--val_config', type=str, help='Path to validation JSON config.')
parser.add_argument('--model_save_folder', default='./model/KITTIDetection', help='Directory to save models.')
parser.add_argument('--checkpoint', type=str, default=None, help='Path to pretrained checkpoint.')
parser.add_argument('--log_folder', default='', help='Directory for logs.')
parser.add_argument("--is_data_pre_quantized", type=bool, default=False, help="Data pre-quantized flag.")
parser.add_argument("--valid_samples", type=str, default='', help="Filter for validation.")

parser.add_argument('--channels', type=int, help='Network channels.', default=32)
parser.add_argument('--kernel_size', type=int, help='Convolution kernel size.', default=3)

parser.add_argument('--batch_size', type=int, help='Batch size.', default=1)
parser.add_argument('--learning_rate', type=float, help='Learning rate.', default=0.0005)
parser.add_argument('--lr_decay', type=float, help='LR decay factor.', default=0.1)
parser.add_argument('--lr_decay_steps', type=int, nargs='+', help='Steps to decay LR.', default=[40000, 90000])
parser.add_argument('--max_steps', type=int, help='Max training steps.', default=110000)
parser.add_argument('--val_interval', type=int, help='Validation interval.', default=500)
parser.add_argument('--log_interval', type=int, help='Logging interval.', default=100)
parser.add_argument('--stage', type=str, help='Model stage.', default="UCM_Attention")

parser.add_argument('--posQ', type=int, help='Quantization scale.', default=1)
parser.add_argument('--use_augmentation', type=bool, help='Use augmentation.', default=False)
parser.add_argument('--shift_range', type=int, help='Shift range.', default=1.0)
parser.add_argument('--use_patch', type=bool, help='Use patch.', default=False)
parser.add_argument('--max_num', type=int, help='Max points per patch.', default=400000)
parser.add_argument('--preprocess_scale', type=float, help='Preprocess scale.', default=1.0)
parser.add_argument('--preprocess_shift', type=float, help='Preprocess shift.', default=0.0)

# WandB arguments
parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases.')
parser.add_argument('--wandb_project', type=str, default='pcc_dense_training', help='W&B project name.')
parser.add_argument('--wandb_name', type=str, default=None, help='W&B run name.')

args = parser.parse_args()

# ======================== Setup Logging ========================

os.makedirs(args.model_save_folder, exist_ok=True)
log_folder = args.log_folder if args.log_folder else args.model_save_folder
os.makedirs(log_folder, exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_folder, f'train_{args.stage}_{timestamp}.log')

logger = logging.getLogger('training')
logger.setLevel(logging.INFO)
if logger.handlers: logger.handlers.clear()

file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info(f"=== Start Training {timestamp} ===")
logger.info(f"Config: {vars(args)}")

if args.use_wandb:
    if wandb is None:
        logger.error("WandB not installed.")
        exit(1)
    run_name = args.wandb_name if args.wandb_name else f"{args.stage}-{timestamp}"
    wandb.init(project=args.wandb_project, name=run_name, config=args)

# ======================== Data Loading ========================

logger.info("--- Preparing DataLoaders ---")

# --- Train DataLoader ---
train_dataset_params = {
    'posQ': args.posQ,
    'is_pre_quantized': args.is_data_pre_quantized,
    'use_augmentation': args.use_augmentation,
    'shift_range': args.shift_range,
    'use_patch': args.use_patch,
    'max_num': args.max_num,
    'preprocess_scale': args.preprocess_scale,
    'preprocess_shift': args.preprocess_shift
}
train_dataset = WeightedMultiDataset(config_path=args.train_config, **train_dataset_params)

train_dataflow = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=args.batch_size,
    collate_fn=custom_collate_fn,
    num_workers=4,
    shuffle=True  
)
logger.info(f"Train loader ready. Steps per epoch: {len(train_dataset)}")

# --- Val DataLoader ---
val_dataflow = None
if args.val_config:
    try:
        with open(args.val_config, 'r') as f:
            val_config = json.load(f)
        
        val_datasets = []
        logger.info(f"Loading validation datasets from '{args.val_config}'...")
        for ds_config in val_config['datasets']:
            all_files = get_files_from_config(ds_config, is_training=False)
            if not all_files: continue
            
            val_params = {
                'posQ': args.posQ,
                'is_pre_quantized': args.is_data_pre_quantized,
                'use_augmentation': False,
                'shift_range': args.shift_range,
                'use_patch': False,
                'max_num': args.max_num,
                'preprocess_scale': args.preprocess_scale,
                'preprocess_shift': args.preprocess_shift,
            }
            
            overrides = {}
            for key, value in ds_config.items():
                if key in val_params:
                    val_params[key] = value
                    overrides[key] = value
            
            sub_dataset = UnifiedPCDataset(all_files, **val_params)
            val_datasets.append(sub_dataset)

        if val_datasets:
            full_val_dataset = ConcatDataset(val_datasets)
            val_dataflow = torch.utils.data.DataLoader(
                dataset=full_val_dataset,
                batch_size=1, 
                collate_fn=sparse_collate_fn,
                num_workers=4,
                shuffle=False
            )
            logger.info(f"Val loader ready. Total samples: {len(full_val_dataset)}")
    except Exception as e:
        logger.error(f"Error loading validation data: {e}", exc_info=True)
else:
    logger.warning("No --val_config provided.")

# ======================== Model Setup ========================

if args.stage == "UCM":
    from model import UCM_Context_Model as Network
elif args.stage == "UCM_1Stage":
    from model import UCM_Context_Model_1Stage as Network
    
net = Network(channels=args.channels, kernel_size=args.kernel_size, device=device).to(device).train()
optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)

# Load Checkpoint
if args.checkpoint and os.path.exists(args.checkpoint):
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
        
        missing, unexpected = net.load_state_dict(state_dict, strict=False)
        if missing: logger.warning(f"Missing keys: {missing}")
        if unexpected: logger.warning(f"Unexpected keys: {unexpected}")
        logger.info("Checkpoint loaded (strict=False).")
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}", exc_info=True)

# Log Model Info
logger.info(f"Model Stage: {args.stage}")
total_params = sum(p.numel() for p in net.parameters())
trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
logger.info(f"Total Params: {total_params:,} | Trainable: {trainable_params:,}")

if args.use_wandb:
    wandb.watch(net, log='all', log_freq=args.log_interval)

best_val_loss = float('inf')
best_model_path = os.path.join(args.model_save_folder, f'best_model_{args.stage}.pt')

def validate():
    if val_dataflow is None: return float('inf')
    net.eval()
    val_losses = []
    with torch.no_grad():
        for data in val_dataflow:
            x = data['input'].to(device=device)
            loss = net(x)
            val_losses.append(loss.item())
    
    avg_val_loss = np.array(val_losses).mean()
    net.train()
    return avg_val_loss

# ======================== Training Loop ========================

losses = []
global_step = 0

logger.info("Starting Training Loop...")

try:
    for epoch in range(1, 9999):
        logger.info(f"Epoch {epoch} started at {datetime.datetime.now()}")
        for data in train_dataflow:
            x = data['input'].to(device=device)
            optimizer.zero_grad()
            
            # Forward pass
            loss = net(x)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            global_step += 1
            losses.append(loss.item())

            # Logging
            if global_step % args.log_interval == 0:
                avg_loss = np.array(losses).mean()
                logger.info(f'Step: {global_step}, Loss: {avg_loss:.5f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
                
                if args.use_wandb:
                    wandb.log({
                        "epoch": epoch,
                        "train_loss": avg_loss,
                        "learning_rate": optimizer.param_groups[0]["lr"]
                    }, step=global_step)
                
                losses = []

            # Validation
            if val_dataflow is not None and global_step % args.val_interval == 0:
                val_loss = validate()
                logger.info(f'Validation | Step:{global_step} | Val Loss:{val_loss:.5f}')
                
                if args.use_wandb:
                    wandb.log({"val_loss": val_loss}, step=global_step)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(net.state_dict(), best_model_path)
                    logger.info(f'New best model saved. Val Loss: {best_val_loss:.5f}')

            # LR Decay
            if global_step in args.lr_decay_steps:
                args.learning_rate = args.learning_rate * args.lr_decay
                for g in optimizer.param_groups:
                    g['lr'] = args.learning_rate
                logger.info(f'LR Decayed to: {args.learning_rate:.6f}')

            # Periodic Save
            if global_step % 500 == 0:
                model_path = os.path.join(args.model_save_folder, f'ckpt_{args.stage}.pt')
                torch.save(net.state_dict(), model_path)
            
            if global_step >= args.max_steps:
                break

        if global_step >= args.max_steps:
            logger.info(f"Max steps {args.max_steps} reached.")
            break

except Exception as e:
    logger.exception(f"Error during training: {str(e)}")
    torch.save(net.state_dict(), os.path.join(args.model_save_folder, f'error_model_{args.stage}.pt'))

finally:
    if val_dataflow is not None:
        try:
            final_val_loss = validate()
            logger.info(f'Final Val Loss: {final_val_loss:.5f} | Best: {best_val_loss:.5f}')
        except Exception as e:
            logger.error(f"Final validation failed: {e}")

    final_model_path = os.path.join(args.model_save_folder, f'final_model_{args.stage}.pt')
    torch.save(net.state_dict(), final_model_path)
    logger.info(f'Final model saved: {final_model_path}')
    
    if args.use_wandb:
        wandb.finish()
        
    logger.info("=== Training Finished ===")