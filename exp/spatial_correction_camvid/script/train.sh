#!/bin/bash
cd /ghome/zhuangjf/git_repo/DAVSS && \
python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=$RANDOM exp/spatial_correction_camvid/python/train.py \
        --exp_name spatial_correction_camvid \
        --root_data_path /gdata/zhuangjf/CamVid \
        --root_gt_path /gdata/zhuangjf/CamVid \
        --train_list_path /ghome/zhuangjf/git_repo/DAVSS/data/list/camvid/trainval.txt \
        --test_list_path /ghome/zhuangjf/git_repo/DAVSS/data/list/camvid/test.txt \
        --train_load_path /gdata1/zhuangjf/git_repo/DAVSS/saved_model/pretrained/pretrained_camvid.pth \
        --lr 1e-4 \
        --cfnet_lr 1e-4 \
        --random_seed 666 \
        --weight_decay 0 \
        --train_batch_size 4 \
        --train_num_workers 4 \
        --test_batch_size 2 \
        --test_num_workers 4 \
        --num_epoch 100 \
        --snap_shot 5 \
        --model_save_path /ghome/zhuangjf/git_repo/DAVSS/saved_model/spatial_correction_camvid \
        --tblog_dir /ghome/zhuangjf/git_repo/DAVSS/tblog/spatial_correction_camvid