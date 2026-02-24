export CUDA_VISIBLE_DEVICES=5
export WANDB_MODE=offline

python src/train.py \
    --train_config ./config/train_dataset_thuman.json \
    --val_config ./config/val_dataset_thuman.json \
    --stage UCM \
    --channels 64 \
    --kernel_size 5 \
    --batch_size 1 \
    --use_patch true \
    --use_augmentation true \
    --max_num 200000 \
    --learning_rate 0.0005 \
    --max_steps 40000 \
    --lr_decay_steps 15000 30000 \
    --val_interval 500 \
    --log_interval 100 \
    --model_save_folder ./checkpoint/ucm_thuman \
    --wandb_project AnyPcc \
    --wandb_name ucm_thuman \
    --use_wandb