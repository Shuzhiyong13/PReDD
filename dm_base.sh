# export CUDA_VISIBLE_DEVICES=0
CUDA_VISIBLE_DEVICES=1 python dm_base.py --dataset=imagenet-fruits --ipc=1 --data_path='/home/user/imagenet' \
    --save_path='./saved_results' --pix_init='noise'
