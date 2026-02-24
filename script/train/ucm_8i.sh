export CUDA_VISIBLE_DEVICES=5
export WANDB_MODE=offline

python src/train.py \
    --train_config ./config/train_dataset_8i.json \
    --val_config ./config/val_dataset_8i.json \
    --stage UCM \
    --channels 64 \
    --kernel_size 5 \
    --batch_size 1 \
    --use_patch true \
    --use_augmentation true \
    --max_num 200000 \
    --learning_rate 0.0005 \
    --max_steps 170000 \
    --lr_decay_steps 100000 150000 \
    --val_interval 500 \
    --log_interval 100 \
    --model_save_folder ./checkpoint/ucm_8i \
    --wandb_project AnyPcc \
    --wandb_name ucm_8i \
    # --use_wandb