export CUDA_VISIBLE_DEVICES=5
export WANDB_MODE=offline

# 8iVFB

# The `fixed_k` parameter can improve reconstruction quality in some cases but not in others; it is disabled by default.

python src/test.py \
    --input_glob /data2/wkl/pcc_data/AnyPcc_lossy_testset/8i \
    --ckpt ./checkpoint/ucm_u/best_model_UCM.pt \
    --lossy_ckpt ./checkpoint/ucm_u_1stage/best_model_UCM_1stage.pt \
    --num_samples 4 \
    --lossy_level 3 \
    --posQ 1 \
    --channels 64 \
    --kernel_size 5 \
    --resultdir ./results_test \
    --prefix test \
    # --fixed_k