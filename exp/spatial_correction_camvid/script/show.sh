#!/bin/bash
cd /ghome/zhuangjf/git_repo/DAVSS && \
python3 exp/spatial_correction_camvid/python/show.py \
            --data_path /gdata/zhuangjf/CamVid \
            --gt_path /gdata/zhuangjf/CamVid \
            --data_list_path /ghome/zhuangjf/git_repo/DAVSS/data/list/camvid/test.txt \
            --save_path /gdata1/zhuangjf/git_repo/DAVSS/result/spatial_correction_camvid \
            --scnet_model /gdata1/zhuangjf/git_repo/DAVSS/saved_model/spatial_correction_camvid/best.pth \
            --num_workers 8