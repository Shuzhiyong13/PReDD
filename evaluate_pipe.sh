#!/bin/bash

# 设置参数
MODE="ours"   # NCFM, mtt, dm, ours, random
SAVED="dm"  # 保存的路径！！！！！
LOG_TYPE="semantic"  # "original" or "semantic"

EXPS=("fruits_10ipc")
DATASETS=("imagenet-fruits")

DISTILL_DIFFUSIONS=(
    "/root/PureDD/sample_results/dit-distillation/dm/imagenet-fruits/fruits_10ipc/imagenet_distill_256_256.pt"
)

# GPUS=(0 1 3 5 6) 
GPUS=(0)  # 每个任务分配的 GPU 编号

for i in ${!DATASETS[@]}; do
    DATASET=${DATASETS[$i]}
    EXP=${EXPS[$i]}
    
    DISTILL_DIFFUSION=${DISTILL_DIFFUSIONS[$i]}

    GPU=${GPUS[$i]}

    # 根据MODE变量决定存储路径
    LOG_DIR="./sample_results/dit-distillation/${SAVED}/${DATASET}/${EXP}"

    LOG_FILE="${LOG_DIR}/log_${LOG_TYPE}.txt"
    mkdir -p "$LOG_DIR"

    echo "Launching: $DATASET / $EXP on GPU $GPU"

    CUDA_VISIBLE_DEVICES=$GPU python evaluate.py \
        --dataset="$DATASET" \
        --data_path="/root/autodl-tmp/imagenet" \
        --distill_path="$DISTILL_DIFFUSION" \
        --test_type="$MODE" \
        --depth=5 \
        --width=128 \
        > "$LOG_FILE" &  # 后台运行
done

wait

echo "All parallel experiments finished."
