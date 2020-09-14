import os
import sys
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.model.flownet import FlowNets
from lib.model.deeplabv3plus import deeplabv3plus
from lib.model.dmnet import DMNet
from lib.model.cfnet import CFNet
from lib.model.warpnet import warp


class SCNet(nn.Module):
    def __init__(self, n_classes=19):
        super(SCNet, self).__init__()
        self.deeplab = deeplabv3plus(n_classes=n_classes)
        self.flownet = FlowNets()
        self.cfnet = CFNet(n_classes=n_classes)
        self.dmnet = DMNet()
        self.warpnet = warp()
        self.semantic_loss = nn.CrossEntropyLoss(ignore_index=255)
        self.cfnet_loss = nn.CrossEntropyLoss(ignore_index=255, reduce=False)
        self.dmnet_loss = nn.BCELoss()

        self.set_fix_deeplab()
        self.set_fix_dmnet()

    def forward(self, img_list, label=None):

        n, c, h, w = img_list[0].shape

        img_1_feat = self.deeplab(img_list[0])
        warp_img = F.upsample(img_list[0], scale_factor=0.25, mode='bilinear', align_corners=True)

        img_2_mask = self.deeplab(img_list[1])
        img_2_mask = F.upsample(img_2_mask, scale_factor=4, mode='bilinear', align_corners=True)
        img_2_mask = torch.argmax(img_2_mask, dim=1)

        loss_semantic = 0.0
        loss_cfnet = 0.0

        flow = self.flownet(torch.cat([img_list[1], img_list[0]], dim=1))
        img_2_feat = self.warpnet(img_1_feat, flow)
        warp_img = self.warpnet(warp_img, flow)

        # semantic loss
        img_2_out_propagate = F.upsample(img_2_feat, scale_factor=4, mode='bilinear', align_corners=True)
        loss_semantic += self.semantic_loss(img_2_out_propagate, img_2_mask)

        # smooth loss
        img_2_down = F.upsample(img_list[1], scale_factor=0.25, mode='bilinear', align_corners=True)
        dm_2 = self.dmnet(warp_img, img_2_down)
        dm_2 = F.interpolate(dm_2, scale_factor=4, mode='bilinear', align_corners=True)

        # cfnet loss
        img_2_feat_cc = self.cfnet(img_list[1])
        img_2_out_cc = F.upsample(img_2_feat_cc, scale_factor=4, mode='bilinear', align_corners=True)
        loss = self.cfnet_loss(img_2_out_cc, img_2_mask)
        loss_cfnet += torch.mean(loss * dm_2)

        img_2_out_merge = img_2_out_propagate * (1-dm_2) + img_2_out_cc*dm_2
        loss_semantic += self.semantic_loss(img_2_out_merge, img_2_mask)

        flow = self.flownet(torch.cat([img_list[2], img_list[1]], dim=1))
        img_3_feat = self.warpnet(img_2_feat, flow)
        warp_img = self.warpnet(warp_img, flow)

        # semantic loss
        img_3_out_propagate = F.upsample(img_3_feat, scale_factor=4, mode='bilinear', align_corners=True)
        loss_semantic += self.semantic_loss(img_3_out_propagate, label)

        # smooth loss
        img_3_down = F.upsample(img_list[2], scale_factor=0.25, mode='bilinear', align_corners=True)
        dm_3 = self.dmnet(warp_img, img_3_down)
        dm_3 = F.interpolate(dm_3, scale_factor=4, mode='bilinear', align_corners=True)

        # cfnet loss
        img_3_feat_cc = self.cfnet(image=img_list[2])
        img_3_out_cc = F.upsample(img_3_feat_cc, scale_factor=4, mode='bilinear', align_corners=True)
        loss = self.cfnet_loss(img_3_out_cc, label)
        loss_cfnet += torch.mean(loss * dm_3)

        img_3_out_merge = img_3_out_propagate * (1-dm_3) + img_3_out_cc*dm_3
        loss_semantic += self.semantic_loss(img_3_out_merge, label)

        loss_semantic /= 4
        loss_semantic = torch.unsqueeze(loss_semantic, 0)

        loss_cfnet /= 2
        loss_cfnet = torch.unsqueeze(loss_cfnet, 0)

        return loss_semantic, loss_cfnet

    def set_fix_deeplab(self):
        for param in self.deeplab.parameters():
            param.requires_grad = False

    def set_fix_dmnet(self):
        for param in self.dmnet.parameters():
            param.requires_grad = False


class SCNet_dmnet(nn.Module):
    # For training DMNet
    def __init__(self, n_classes=19):
        super(SCNet_dmnet, self).__init__()
        self.deeplab = deeplabv3plus(n_classes=n_classes)
        self.flownet = FlowNets()
        self.dmnet = DMNet()
        self.warpnet = warp()
        self.dmnet_loss = nn.BCELoss()

        self.set_fix_deeplab()
        self.set_fix_flownet()

    def forward(self, img_list):

        n, c, h, w = img_list[0].shape

        img_1_feat = self.deeplab(img_list[0])
        warp_im = F.upsample(img_list[0], scale_factor=0.25, mode='bilinear', align_corners=True)

        img_2_mask = self.deeplab(img_list[1])
        img_2_mask = F.upsample(img_2_mask, scale_factor=4, mode='bilinear', align_corners=True)
        img_2_mask = torch.argmax(img_2_mask, dim=1)
        img_3_mask = self.deeplab(img_list[2])
        img_3_mask = F.upsample(img_3_mask, scale_factor=4, mode='bilinear', align_corners=True)
        img_3_mask = torch.argmax(img_3_mask, dim=1)

        loss_dmnet = 0.0

        flow = self.flownet(torch.cat([img_list[1], img_list[0]], dim=1))
        img_2_feat = self.warpnet(img_1_feat, flow)
        warp_im = self.warpnet(warp_im, flow)

        img_2_out_propagate = F.upsample(img_2_feat, scale_factor=4, mode='bilinear', align_corners=True)
        img_2_out_propagate = torch.argmax(img_2_out_propagate, dim=1, keepdims=True)

        img_2_down = F.upsample(img_list[1], scale_factor=0.25, mode='bilinear', align_corners=True)
        dm_2 = self.dmnet(warp_im, img_2_down)
        dm_2 = F.interpolate(dm_2, scale_factor=4, mode='bilinear', align_corners=True)
        label_2 = (img_2_out_propagate != img_2_mask.unsqueeze(1)).float().detach()
        loss_dmnet += self.dmnet_loss(dm_2, label_2)

        flow = self.flownet(torch.cat([img_list[2], img_list[1]], dim=1))
        img_3_feat = self.warpnet(img_2_feat, flow)
        warp_im = self.warpnet(warp_im, flow)

        img_3_out_propagate = F.upsample(img_3_feat, scale_factor=4, mode='bilinear', align_corners=True)
        img_3_out_propagate = torch.argmax(img_3_out_propagate, dim=1, keepdims=True)

        img_3_down = F.upsample(img_list[2], scale_factor=0.25, mode='bilinear', align_corners=True)
        dm_3 = self.dmnet(warp_im, img_3_down)
        dm_3 = F.interpolate(dm_3, scale_factor=4, mode='bilinear', align_corners=True)
        label_3 = (img_3_out_propagate != img_3_mask.unsqueeze(1)).float().detach()
        loss_dmnet += self.dmnet_loss(dm_3, label_3)

        loss_dmnet /= 2
        loss_dmnet = torch.unsqueeze(loss_dmnet, 0)

        return loss_dmnet

    def set_fix_deeplab(self):
        for param in self.deeplab.parameters():
            param.requires_grad = False

    def set_fix_flownet(self):
        for param in self.flownet.parameters():
            param.requires_grad = False


# class SCNet_Camvid(nn.Module):
#     def __init__(self, n_classes=19):
#         super(SCNet_Camvid, self).__init__()
#         self.deeplab = deeplabv3plus(n_classes=n_classes)
#         self.flownet = FlowNets()
#         self.cfnet = CFNet(n_classes=n_classes)
#         self.dmnet = DMNet()
#         self.warpnet = warp()
#         self.semantic_loss = nn.CrossEntropyLoss(ignore_index=255)
#         self.cfnet_loss = nn.CrossEntropyLoss(ignore_index=255, reduce=False)
#         self.dmnet_loss = nn.BCELoss()

#         self.set_fix_deeplab()
#         self.set_fix_dmnet()

#     def forward(self, img_1, img_2, img_3, label):

#         n, c, h, w = img_1.shape

#         img_1_feat = self.deeplab(img_1)
#         warp_img = F.upsample(img_1, scale_factor=0.25, mode='bilinear', align_corners=True)

#         img_2_mask = self.deeplab(img_2)
#         img_2_mask = F.upsample(img_2_mask, scale_factor=4, mode='bilinear', align_corners=True)
#         img_2_mask = torch.argmax(img_2_mask, dim=1)

#         loss_semantic = 0.0
#         loss_cfnet = 0.0

#         flow = self.flownet(torch.cat([img_2, img_1], dim=1))
#         img_2_feat = self.warpnet(img_1_feat, flow)
#         warp_img = self.warpnet(warp_img, flow)

#         # semantic loss
#         img_2_out_propagate = F.upsample(img_2_feat, scale_factor=4, mode='bilinear', align_corners=True)
#         loss_semantic += self.semantic_loss(img_2_out_propagate, img_2_mask)

#         # smooth loss
#         img_2_down = F.upsample(img_2, scale_factor=0.25, mode='bilinear', align_corners=True)
#         dm_2 = self.dmnet(warp_img, img_2_down)
#         dm_2 = F.interpolate(dm_2, scale_factor=4, mode='bilinear', align_corners=True)

#         # cfnet loss
#         img_2_feat_cc = self.cfnet(img_2)
#         img_2_out_cc = F.upsample(img_2_feat_cc, scale_factor=4, mode='bilinear', align_corners=True)
#         loss = self.cfnet_loss(img_2_out_cc, img_2_mask)
#         loss_cfnet += torch.mean(loss * dm_2)

#         img_2_out_merge = img_2_out_propagate * (1-dm_2) + img_2_out_cc*dm_2
#         loss_semantic += self.semantic_loss(img_2_out_merge, img_2_mask)

#         flow = self.flownet(torch.cat([img_3, img_2], dim=1))
#         img_3_feat = self.warpnet(img_2_feat, flow)
#         warp_img = self.warpnet(warp_img, flow)

#         # semantic loss
#         img_3_out_propagate = F.upsample(img_3_feat, scale_factor=4, mode='bilinear', align_corners=True)
#         loss_semantic += self.semantic_loss(img_3_out_propagate, label)

#         # smooth loss
#         img_3_down = F.upsample(img_3, scale_factor=0.25, mode='bilinear', align_corners=True)
#         dm_3 = self.dmnet(warp_img, img_3_down)
#         dm_3 = F.interpolate(dm_3, scale_factor=4, mode='bilinear', align_corners=True)

#         # cfnet loss
#         img_3_feat_cc = self.cfnet(image=img_3)
#         img_3_out_cc = F.upsample(img_3_feat_cc, scale_factor=4, mode='bilinear', align_corners=True)
#         loss = self.cfnet_loss(img_3_out_cc, label)
#         loss_cfnet += torch.mean(loss * dm_3)

#         img_3_out_merge = img_3_out_propagate * (1-dm_3) + img_3_out_cc*dm_3
#         loss_semantic += self.semantic_loss(img_3_out_merge, label)

#         loss_semantic /= 4
#         loss_semantic = torch.unsqueeze(loss_semantic, 0)

#         loss_cfnet /= 2
#         loss_cfnet = torch.unsqueeze(loss_cfnet, 0)

#         return loss_semantic, loss_cfnet

#     def set_fix_deeplab(self):
#         for param in self.deeplab.parameters():
#             param.requires_grad = False

#     def set_fix_dmnet(self):
#         for param in self.dmnet.parameters():
#             param.requires_grad = False

if __name__ == '__main__':

    net = SCNet()
    net.cuda().eval()

    img_1 = torch.rand([2, 3, 512, 1024]).cuda()
    img_1_mask = torch.zeros([2, 512, 1024]).long().cuda()
    img_2 = torch.rand([2, 3, 512, 1024]).cuda()
    img_2_mask = torch.zeros([2, 512, 1024]).long().cuda()
    img_3 = torch.rand([2, 3, 512, 1024]).cuda()
    img_3_mask = torch.zeros([2, 512, 1024]).long().cuda()
    label = torch.zeros([2, 512, 1024]).long().cuda()

    with torch.no_grad():
        loss_semantic, loss_cfnet = net(img_1, img_1_mask, img_2, img_2_mask, img_3, img_3_mask, label)
        print(loss_semantic.item(), loss_cfnet.item())
