#!/bin/bash
cd /ghome/zhuangjf/git_repo/DAVSS && \
python3 exp/spatial_correction_cityscapes/python/test_PDA.py \
            --data_path /gpub \
            --gt_path /gdata/zhuangjf/cityscapes/original \
            --data_list_path /ghome/zhuangjf/git_repo/DAVSS/data/list/cityscapes/val.txt \
            --scnet_model /gdata1/zhuangjf/git_repo/DAVSS/saved_model1/spatial_correction_cityscapes/best.pth \
            --num_workers 8