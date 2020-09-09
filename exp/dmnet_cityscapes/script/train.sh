#!/bin/bash
cd /ghome/zhuangjf/git_repo/DAVSS && \
python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=$RANDOM exp/dmnet_cityscapes/python/train.py \
        --exp_name dmnet_cityscapes \
        --root_data_path /gpub \
        --root_gt_path /gdata/zhuangjf/cityscapes/original \
        --root_mask_path /gdata1/zhuangjf/DAVSS/result1/deeplab_results \
        --train_list_path /ghome/zhuangjf/git_repo/DAVSS/data/list/cityscapes/train.txt \
        --test_list_path /ghome/zhuangjf/git_repo/DAVSS/data/list/cityscapes/val.txt \
        --train_load_path /gdata1/zhuangjf/git_repo/DAVSS/saved_model/pretrained/pretrained_cityscapes.pth \
        --dmnet_lr 1e-4 \
        --random_seed 666 \
        --weight_decay 0 \
        --train_batch_size 12 \
        --train_num_workers 4 \
        --test_batch_size 2 \
        --test_num_workers 4 \
        --num_epoch 100 \
        --snap_shot 5 \
        --model_save_path /ghome/zhuangjf/git_repo/DAVSS/saved_model/dmnet_cityscapes \
        --tblog_dir /ghome/zhuangjf/git_repo/DAVSS/tblog/dmnet_cityscapes