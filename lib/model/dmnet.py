import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


class DMNet(nn.Module):
    def __init__(self):

        super(DMNet, self).__init__()

        self.conv1 = nn.Sequential(SeparableConv2d(3, 4, kernel_size=3, stride=1, padding=1, bias=True),
                                   nn.SyncBatchNorm(4), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(SeparableConv2d(4, 8, kernel_size=3, stride=1, padding=1, bias=True),
                                   nn.SyncBatchNorm(8), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(SeparableConv2d(8, 16, kernel_size=3, stride=1, padding=1, bias=True),
                                   nn.SyncBatchNorm(16), nn.ReLU(inplace=True))
        self.conv4 = nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0, bias=True)

        self.weight_init()

    def forward(self, feat1, feat2):
        feat1 = self.conv1(feat1)
        feat1 = self.conv2(feat1)
        feat1 = self.conv3(feat1)
        feat1 = self.conv4(feat1)

        feat2 = self.conv1(feat2)
        feat2 = self.conv2(feat2)
        feat2 = self.conv3(feat2)
        feat2 = self.conv4(feat2)

        diff = -F.cosine_similarity(feat1, feat2, dim=1)
        diff = diff*0.5 + 0.5
        diff = torch.clamp(diff, min=0.0, max=1.0)
        diff = diff.unsqueeze(1)
        return diff

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init.xavier_normal_(m.weight)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.SyncBatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.fill_(0)


class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=0, dilation=1, bias=False, separable=True):
        super(SeparableConv2d, self).__init__()
        self.separable = separable
        if self.separable:
            self.depthwise = nn.Conv2d(inplanes,
                                       inplanes,
                                       kernel_size,
                                       stride,
                                       padding,
                                       dilation,
                                       groups=inplanes,
                                       bias=bias)
            self.depthwise_bn = nn.SyncBatchNorm(inplanes, eps=1e-05, momentum=0.0003)
            self.depthwise_relu = nn.ReLU(inplace=True)
            self.pointwise = nn.Conv2d(inplanes,
                                       planes,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0,
                                       dilation=1,
                                       groups=1,
                                       bias=bias)
            self.pointwise_bn = nn.SyncBatchNorm(planes, eps=1e-05, momentum=0.0003)
            self.pointwise_relu = nn.ReLU(inplace=True)
        else:
            self.conv = nn.Conv2d(inplanes, planes, kernel_size, stride, padding, dilation, bias=bias)
            self.bn = nn.SyncBatchNorm(planes, eps=1e-05, momentum=0.0003)
            self.relu = nn.ReLU(inplace=True)

        self._init_weight()

    def forward(self, x):
        if self.separable:
            x = self.depthwise(x)
            x = self.depthwise_bn(x)
            x = self.depthwise_relu(x)
            x = self.pointwise(x)
            x = self.pointwise_bn(x)
            x = self.pointwise_relu(x)

        else:
            x = self.conv(x)
            x = self.bn(x)
            x = self.relu(x)
        return x

    def _init_weight(self):
        if self.separable:
            torch.nn.init.normal_(self.depthwise.weight, std=0.33)
            self.depthwise_bn.weight.data.fill_(1)
            self.depthwise_bn.bias.data.zero_()
            torch.nn.init.normal_(self.pointwise.weight, std=0.33)
            self.pointwise_bn.weight.data.fill_(1)
            self.pointwise_bn.bias.data.zero_()
        else:
            torch.nn.init.normal_(self.conv.weight, std=0.33)
            self.bn.weight.data.fill_(1)
            self.bn.bias.data.zero_()


if __name__ == '__main__':
    model = DMNet()
    model.cuda().eval()

    feat1 = torch.rand([1, 3, 256, 512]).cuda()
    feat2 = torch.rand([1, 3, 256, 512]).cuda()

    with torch.no_grad():
        diff_map = model(feat1, feat2)
        print(diff_map.shape)