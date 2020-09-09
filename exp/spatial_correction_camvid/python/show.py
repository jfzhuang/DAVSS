import os
import sys
import cv2
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from lib.model.scnet import SCNet
from lib.dataset.camvid import camvid_video_dataset_PDA
from lib.dataset.utils import decode_labels_camvid


def get_arguments():
    parser = argparse.ArgumentParser(description="Test the SCNet")
    ###### general setting ######
    parser.add_argument("--data_list_path", type=str, help="path to the data list")
    parser.add_argument("--data_path", type=str, help="path to the data")
    parser.add_argument("--gt_path", type=str, help="path to the ground truth")
    parser.add_argument("--save_path", type=str, help="path to save results")
    parser.add_argument("--scnet_model", type=str, help="path to the trained SCNet model")

    ###### inference setting ######
    parser.add_argument("--num_workers", type=int, help="num of cpus used")

    return parser.parse_args()


def test():
    args = get_arguments()
    print(args)

    net = SCNet(n_classes=11)
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

    test_data = camvid_video_dataset_PDA(args.data_path, args.gt_path, args.data_list_path)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=args.num_workers)

    distance = 10

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    with torch.no_grad():
        for d in range(9, distance):
            for step, sample in enumerate(test_data_loader):
                print(step)

                img_list, gt_label = sample

                result_list = []

                img = img_list[9 - d].cuda()
                feat = deeplab(img)
                warp_im = F.upsample(img, scale_factor=0.25, mode='bilinear', align_corners=True)

                feat_rgb = pred2im(feat)
                feat_rgb = cv2.resize(feat_rgb, (480, 360))
                result_list.append(feat_rgb)

                for i in range(d):
                    img_1 = img_list[9 - d + i].cuda()
                    img_2 = img_list[10 - d + i].cuda()
                    flow = flownet(torch.cat([img_2, img_1], dim=1))
                    feat = warpnet(feat, flow)
                    warp_im = warpnet(warp_im, flow)

                    img_2_down = F.upsample(img_2, scale_factor=0.25, mode='bilinear', align_corners=True)
                    dm = dmnet(warp_im, img_2_down)
                    dm_up = F.interpolate(dm, scale_factor=4, mode='bilinear', align_corners=True)

                    feat_cc = cfnet(img_2)
                    feat_cc_up = F.interpolate(feat_cc, scale_factor=4, mode='bilinear', align_corners=True)

                    feat_up = F.interpolate(feat, scale_factor=4, mode='bilinear', align_corners=True)
                    feat_merge = feat_up * (1-dm_up) + feat_cc_up*dm_up

                    feat_merge_rgb = pred2im(feat_merge)
                    feat_merge_rgb = cv2.resize(feat_merge_rgb, (480, 360))
                    result_list.append(feat_merge_rgb)

                gt_rgb = gt_label.clone()
                gt_rgb = decode_labels_camvid(gt_rgb.squeeze())
                gt_rgb = cv2.resize(gt_rgb, (480, 360))
                result_list.append(gt_rgb)

                zeros = np.zeros([360, 480, 3], dtype=np.uint8)
                result_list.append(zeros)

                result_1 = np.concatenate(result_list[:4], axis=1)
                result_2 = np.concatenate(result_list[4:8], axis=1)
                result_3 = np.concatenate(result_list[8:], axis=1)
                result = np.concatenate([result_1, result_2, result_3], axis=0)
                cv2.imwrite(os.path.join(args.save_path, '{}.png'.format(step)), result)

                if step == 20:
                    break


def pred2im(pred):
    pred = torch.argmax(pred, dim=1).squeeze().cpu().numpy()
    pred = decode_labels_camvid(pred)
    return pred


if __name__ == '__main__':
    test()
