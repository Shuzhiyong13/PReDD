#!/bin/bash 
# Parameter settings
MODE="dm"  # Used for save path and distill_type
EXPS=("fruits_10ipc")
DATASETS=("imagenet-fruits")
DISTILL_PATHS=(
    "/root/PReDD/saved_results/dm/imagenet-fruits/10_ipc/res_128_DM_imagenet-fruits_ConvNet_10ipc.pt"
)

# GPU device
GPU=0

# Loop through experiments
for i in "${!EXPS[@]}"; do
    EXP=${EXPS[$i]}
    DATASET=${DATASETS[$i]}
    DISTILL_PATH=${DISTILL_PATHS[$i]}

    # Create save directory
    SAVE_DIR="./sample_results/dit-distillation/${MODE}"

    echo "Launching distillation with:"
    echo "MODE: $MODE"
    echo "DATASET: $DATASET"
    echo "EXP: $EXP"
    echo "GPU: $GPU"
    echo "DISTILL_PATH: $DISTILL_PATH"

    CUDA_VISIBLE_DEVICES=$GPU python predd.py \
        --model DiT-XL/2 \
        --image-size 256 \
        --ckpt pretrained_models/DiT-XL-2-256x256.pt \
        --save-dir "$SAVE_DIR" \
        --exp "$EXP" \
        --dataset "$DATASET" \
        --num-sampling-steps 50 \
        --forward-t 20 \
        --reverse-t 20 \
        --diff-loop 1 \
        --seed 1 \
        --batch-size 1 \
        --saved True \
        --save_origin True \
        --distill-path "$DISTILL_PATH" \
        --distill-type "$MODE" \
        --cfg-scale 4

    echo "Experiment $EXP finished!"
done

# # 参数设置, 请不要忘记修改!
# MODE="NCFM"  # 用于保存路径 和 distill_type
# EXPS=( "cats_10ipc")
# DATASETS=("imagenet-cats")
# DISTILL_PATHS=(
#     "/root/PReDD/saved_results/NCFM/images-meow10/data_20000.pt"
# )
# NCFM 相关超参数请参考 github README