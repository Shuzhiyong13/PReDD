#!/bin/bash

# 参数设置
MODE="dm"  # 用于保存路径和 distill_type

# GPU 设备
GPU=0

EXPS=( "fruits_10ipc")
DATASETS=("imagenet-fruits")
DISTILL_PATHS=(
    "/root/PReDD/saved_results/dm/imagenet-fruits/10_ipc/res_128_DM_imagenet-fruits_ConvNet_10ipc.pt"
)

# 循环执行
for i in "${!EXPS[@]}"; do
    EXP=${EXPS[$i]}
    DATASET=${DATASETS[$i]}
    DISTILL_PATH=${DISTILL_PATHS[$i]}

    # 构建保存目录（保持原逻辑）
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
