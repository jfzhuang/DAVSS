#!/bin/bash
cd /ghome/zhuangjf/git_repo/DAVSS && \
python3 exp/spatial_correction_camvid/python/test_PDA.py \
            --data_path /gdata/zhuangjf/CamVid \
            --gt_path /gdata/zhuangjf/CamVid \
            --data_list_path /ghome/zhuangjf/git_repo/DAVSS/data/list/camvid/test.txt \
            --scnet_model /gdata1/zhuangjf/git_repo/DAVSS/saved_model/spatial_correction_camvid/best.pth \
            --num_workers 8