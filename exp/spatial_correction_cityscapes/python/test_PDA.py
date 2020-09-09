import os
import sys
import cv2
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.model.scnet import SCNet
from lib.dataset.cityscapes import cityscapes_video_dataset_PDA
from lib.dataset.utils import runningScore


def get_arguments():
    parser = argparse.ArgumentParser(description="Test the SCNet")
    ###### general setting ######
    parser.add_argument("--data_list_path", type=str, help="path to the data list")
    parser.add_argument("--data_path", type=str, help="path to the data")
    parser.add_argument("--gt_path", type=str, help="path to the ground truth")
    parser.add_argument("--scnet_model", type=str, help="path to the trained scnet model")

    ###### inference setting ######
    parser.add_argument("--num_workers", type=int, help="num of cpus used")

    return parser.parse_args()


def test():
    args = get_arguments()
    print(args)

    net = SCNet()
    old_weight = torch.load(args.scnet_model)
    new_weight = {}
    for k, v in old_weight.items():
        new_k = k.replace('module.', '')
        new_weight[new_k] = v
    net.load_state_dict(new_weight, strict=True)
    net.cuda().eval()

    deeplab = net.deeplab
    flownet = net.flownet
    cfnet = net.cfnet
    dmnet = net.dmnet
    warpnet = net.warpnet

    test_data = cityscapes_video_dataset_PDA(args.data_path, args.gt_path, args.data_list_path)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=args.num_workers)

    miou_cal = runningScore(n_classes=19)
    distance = 10

    with torch.no_grad():
        for d in range(1, distance):
            for step, sample in enumerate(test_data_loader):
                img_list, gt_label = sample
                gt_label = gt_label.squeeze().cpu().numpy()

                img = img_list[9 - d].cuda()
                feat = deeplab(img)
                warp_im = F.upsample(img, scale_factor=0.25, mode='bilinear', align_corners=True)

                for i in range(d):
                    img_1 = img_list[9 - d + i].cuda()
                    img_2 = img_list[10 - d + i].cuda()
                    flow = flownet(torch.cat([img_2, img_1], dim=1))
                    feat = warpnet(feat, flow)
                    warp_im = warpnet(warp_im, flow)

                feat = F.interpolate(feat, scale_factor=4, mode='bilinear', align_corners=True)

                img_2_down = F.upsample(img_2, scale_factor=0.25, mode='bilinear', align_corners=True)
                dm = dmnet(warp_im, img_2_down)
                dm = F.interpolate(dm, scale_factor=4, mode='bilinear', align_corners=True)

                feat_cc = cfnet(img_2)
                feat_cc = F.interpolate(feat_cc, scale_factor=4, mode='bilinear', align_corners=True)

                feat_merge = feat * (1-dm) + feat_cc*dm
                out = torch.argmax(feat_merge, dim=1)
                out = out.squeeze().cpu().numpy()
                miou_cal.update(gt_label, out)

                # if step == 5:
                #     break

            miou, iou = miou_cal.get_scores(return_class=True)
            miou_cal.reset()
            print('distance:{} miou:{}'.format(d, miou))
            print('class iou:')
            for i in range(len(iou)):
                print(iou[i])


if __name__ == '__main__':
    test()
