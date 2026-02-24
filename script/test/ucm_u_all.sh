export CUDA_VISIBLE_DEVICES=5
export WANDB_MODE=offline

# 8iVFB
python src/test.py \
    --input_glob /data2/wkl/pcc_data/Cat3A/redandblack \
    --ckpt ./checkpoint/ucm_u/best_model_UCM.pt \
    --posQ 1 \
    --channels 64 \
    --kernel_size 5 \
    --num_samples -1 \
    --resultdir ./results_test/unimodel \
    --prefix redandblack \
    --compress_only \

python src/test.py \
    --input_glob /data2/wkl/pcc_data/Cat3A/loot \
    --ckpt ./checkpoint/ucm_u/best_model_UCM.pt \
    --posQ 1 \
    --channels 64 \
    --kernel_size 5 \
    --num_samples -1 \
    --resultdir ./results_test/unimodel \
    --prefix loot \
    --compress_only \

# MVUB
python src/test.py \
    --input_glob /data2/wkl/pcc_data/MVUB/ricardo10 \
    --ckpt ./checkpoint/ucm_u/best_model_UCM.pt \
    --posQ 1 \
    --channels 64 \
    --kernel_size 5 \
    --num_samples -1 \
    --resultdir ./results_test/unimodel \
    --prefix ricardo10 \
    --compress_only \

python src/test.py \
    --input_glob /data2/wkl/pcc_data/MVUB/phil10 \
    --ckpt ./checkpoint/ucm_u/best_model_UCM.pt \
    --posQ 1 \
    --channels 64 \
    --kernel_size 5 \
    --num_samples -1 \
    --resultdir ./results_test/unimodel \
    --prefix phil10 \
    --compress_only \

# OWlii
python src/test.py \
    --input_glob /data2/wkl/pcc_data/Owlii/basketball_player_vox11 \
    --ckpt ./checkpoint/ucm_u/best_model_UCM.pt \
    --posQ 1 \
    --channels 64 \
    --kernel_size 5 \
    --num_samples -1 \
    --resultdir ./results_test/unimodel \
    --prefix basketball_player_vox11 \
    --compress_only \

python src/test.py \
    --input_glob /data2/wkl/pcc_data/Owlii/model_vox11 \
    --ckpt ./checkpoint/ucm_u/best_model_UCM.pt \
    --posQ 1 \
    --channels 64 \
    --kernel_size 5 \
    --num_samples -1 \
    --resultdir ./results_test/unimodel \
    --prefix model_vox11 \
    --compress_only \

python src/test.py \
    --input_glob /data2/wkl/pcc_data/Owlii/exercise_vox11 \
    --ckpt ./checkpoint/ucm_u/best_model_UCM.pt \
    --posQ 1 \
    --channels 64 \
    --kernel_size 5 \
    --num_samples -1 \
    --resultdir ./results_test/unimodel \
    --prefix exercise_vox11 \
    --compress_only \

python src/test.py \
    --input_glob /data2/wkl/pcc_data/Owlii/dancer_vox11 \
    --ckpt ./checkpoint/ucm_u/best_model_UCM.pt \
    --posQ 1 \
    --channels 64 \
    --kernel_size 5 \
    --num_samples -1 \
    --resultdir ./results_test/unimodel \
    --prefix dancer_vox11 \
    --compress_only \

# Thuman
python src/test.py \
    --input_glob /data2/wkl/pcc_data/thuman \
    --ckpt ./checkpoint/ucm_u/best_model_UCM.pt \
    --posQ 1 \
    --channels 64 \
    --kernel_size 5 \
    --num_samples -1 \
    --resultdir ./results_test/unimodel \
    --prefix thuman \
    --compress_only \

# scannet
python src/test.py \
    --input_glob /data2/wkl/pcc_data/AnyPcc_lossy_testset/scans_test_quan_dupli \
    --ckpt ./checkpoint/ucm_u/best_model_UCM.pt \
    --posQ 1 \
    --channels 64 \
    --kernel_size 5 \
    --num_samples -1 \
    --resultdir ./results_test/unimodel \
    --prefix scannet \
    --compress_only \

# kitti
python src/test.py \
    --input_glob /data2/wkl/pcc_data/AnyPcc_testset/kitti_test_quan \
    --ckpt ./checkpoint/ucm_u/best_model_UCM.pt \
    --posQ 16 \
    --channels 64 \
    --kernel_size 5 \
    --num_samples -1 \
    --resultdir ./results_test/unimodel \
    --prefix kitti_q16 \
    --compress_only \

# ford
python src/test.py \
    --input_glob /data2/wkl/pcc_data/AnyPcc_testset/ford_test \
    --ckpt ./checkpoint/ucm_u/best_model_UCM.pt \
    --posQ 16 \
    --channels 64 \
    --kernel_size 5 \
    --num_samples -1 \
    --resultdir ./results_test/unimodel \
    --prefix ford_q16 \
    --compress_only \

# GS
python src/test.py \
    --input_glob /data2/wkl/pcc_data/gs_pcc/testset/quantized \
    --ckpt ./checkpoint/ucm_u/best_model_UCM.pt \
    --posQ 1 \
    --channels 64 \
    --kernel_size 5 \
    --num_samples -1 \
    --resultdir ./results_test/unimodel \
    --prefix gs \
    --compress_only \