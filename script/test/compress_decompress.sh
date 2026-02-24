export CUDA_VISIBLE_DEVICES=5
export WANDB_MODE=offline

# GS
python src/compress.py \
    --input_glob /data2/wkl/pcc_data/gs_pcc/testset/quantized \
    --ckpt ./checkpoint/ucm_u/best_model_UCM.pt \
    --posQ 1 \
    --channels 64 \
    --kernel_size 5 \
    --num_samples -1 \
    --output_dir ./bins \


python src/decompress.py \
    --input_glob ./bins \
    --ckpt ./checkpoint/ucm_u/best_model_UCM.pt \
    --posQ 1 \
    --channels 64 \
    --kernel_size 5 \
    --num_samples -1 \
    --output_dir ./decoded \