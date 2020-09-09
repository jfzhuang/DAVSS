import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class deeplabv3plus(nn.Module):
    def __init__(self, n_classes=21):
        super(deeplabv3plus, self).__init__()

        # Atrous Conv
        self.xception_features = Xception()
        # ASPP
        rates = [1, 6, 12, 18]

        self.aspp0 = ASPP_module(2048, 256, rate=rates[0], separable=False)
        self.aspp1 = ASPP_module(2048, 256, rate=rates[1])
        self.aspp2 = ASPP_module(2048, 256, rate=rates[2])
        self.aspp3 = ASPP_module(2048, 256, rate=rates[3])

        self.image_pooling = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                                           nn.SyncBatchNorm(256, eps=1e-05, momentum=0.0003), nn.ReLU(inplace=True))

        self.concat_projection = nn.Sequential(nn.Conv2d(1280, 256, 1, stride=1, bias=False),
                                               nn.SyncBatchNorm(256, eps=1e-05, momentum=0.0003), nn.ReLU(inplace=True),
                                               nn.Dropout2d(p=0.1))

        # adopt [1x1, 48] for channel reduction.
        self.feature_projection0_conv = nn.Conv2d(256, 48, 1, bias=False)
        self.feature_projection0_bn = nn.SyncBatchNorm(48, eps=1e-03, momentum=0.0003)
        self.feature_projection0_relu = nn.ReLU(inplace=True)

        self.decoder = nn.Sequential(SeparableConv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                     SeparableConv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False))
        self.logits = nn.Conv2d(256, n_classes, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, image):
        n, c, h, w = image.shape
        x, low_level_features = self.xception_features(image)

        x1 = self.aspp0(x)
        x2 = self.aspp1(x)
        x3 = self.aspp2(x)
        x4 = self.aspp3(x)
        x5 = self.image_pooling(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x5, x1, x2, x3, x4), dim=1)
        x = self.concat_projection(x)
        x = F.interpolate(x,
                          size=(int(math.ceil(image.size()[-2] / 4)), int(math.ceil(image.size()[-1] / 4))),
                          mode='bilinear',
                          align_corners=True)

        low_level_features = self.feature_projection0_conv(low_level_features)
        low_level_features = self.feature_projection0_bn(low_level_features)
        low_level_features = self.feature_projection0_relu(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)
        x = self.decoder(x)
        x = self.logits(x)

        return x


class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, rate, separable=True):
        super(ASPP_module, self).__init__()
        if rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = rate
        self.atrous_convolution = SeparableConv2d(inplanes,
                                                  planes,
                                                  kernel_size=kernel_size,
                                                  stride=1,
                                                  padding=padding,
                                                  dilation=rate,
                                                  bias=False,
                                                  separable=separable)

    def forward(self, x):
        x = self.atrous_convolution(x)
        return x


class Xception(nn.Module):
    """
    Modified Alighed Xception
    """
    def __init__(self, inplanes=3, os=16):
        super(Xception, self).__init__()

        if os == 16:
            entry_block3_stride = 2
            middle_block_rate = 1
            exit_block_rates = (1, 2)
        elif os == 8:
            entry_block3_stride = 1
            middle_block_rate = 2
            exit_block_rates = (2, 4)
        else:
            raise NotImplementedError

        # Entry flow
        self.entry_flow_conv1_1 = resnet_utils_conv2d_same(inplanes, 32, 3, stride=2, bias=False)
        self.entry_flow_bn1_1 = nn.SyncBatchNorm(32, eps=1e-03, momentum=0.0003)
        self.entry_flow_relu1_1 = nn.ReLU(inplace=True)

        self.entry_flow_conv1_2 = resnet_utils_conv2d_same(32, 64, 3, stride=1, bias=False)
        self.entry_flow_bn1_2 = nn.SyncBatchNorm(64, eps=1e-03, momentum=0.0003)
        self.entry_flow_relu1_2 = nn.ReLU(inplace=True)

        self.entry_flow_block1_unit_1 = xception_module(inplanes=64,
                                                        depth_list=[128, 128, 128],
                                                        skip_connection_type='conv',
                                                        activation_fn_in_separable_conv=False,
                                                        stride=2)
        self.entry_flow_block2_unit_1 = xception_module(inplanes=128,
                                                        depth_list=[256, 256, 256],
                                                        skip_connection_type='conv',
                                                        activation_fn_in_separable_conv=False,
                                                        stride=2,
                                                        low_level_features=True)
        self.entry_flow_block3_unit_1 = xception_module(inplanes=256,
                                                        depth_list=[728, 728, 728],
                                                        skip_connection_type='conv',
                                                        activation_fn_in_separable_conv=False,
                                                        stride=2)

        # Middle flow
        self.middle_flow_block1_unit_1 = xception_module(inplanes=728,
                                                         depth_list=[728, 728, 728],
                                                         skip_connection_type='sum',
                                                         activation_fn_in_separable_conv=False,
                                                         stride=1)
        self.middle_flow_block1_unit_2 = xception_module(inplanes=728,
                                                         depth_list=[728, 728, 728],
                                                         skip_connection_type='sum',
                                                         activation_fn_in_separable_conv=False,
                                                         stride=1)
        self.middle_flow_block1_unit_3 = xception_module(inplanes=728,
                                                         depth_list=[728, 728, 728],
                                                         skip_connection_type='sum',
                                                         activation_fn_in_separable_conv=False,
                                                         stride=1)
        self.middle_flow_block1_unit_4 = xception_module(inplanes=728,
                                                         depth_list=[728, 728, 728],
                                                         skip_connection_type='sum',
                                                         activation_fn_in_separable_conv=False,
                                                         stride=1)
        self.middle_flow_block1_unit_5 = xception_module(inplanes=728,
                                                         depth_list=[728, 728, 728],
                                                         skip_connection_type='sum',
                                                         activation_fn_in_separable_conv=False,
                                                         stride=1)
        self.middle_flow_block1_unit_6 = xception_module(inplanes=728,
                                                         depth_list=[728, 728, 728],
                                                         skip_connection_type='sum',
                                                         activation_fn_in_separable_conv=False,
                                                         stride=1)
        self.middle_flow_block1_unit_7 = xception_module(inplanes=728,
                                                         depth_list=[728, 728, 728],
                                                         skip_connection_type='sum',
                                                         activation_fn_in_separable_conv=False,
                                                         stride=1)
        self.middle_flow_block1_unit_8 = xception_module(inplanes=728,
                                                         depth_list=[728, 728, 728],
                                                         skip_connection_type='sum',
                                                         activation_fn_in_separable_conv=False,
                                                         stride=1)
        self.middle_flow_block1_unit_9 = xception_module(inplanes=728,
                                                         depth_list=[728, 728, 728],
                                                         skip_connection_type='sum',
                                                         activation_fn_in_separable_conv=False,
                                                         stride=1)
        self.middle_flow_block1_unit_10 = xception_module(inplanes=728,
                                                          depth_list=[728, 728, 728],
                                                          skip_connection_type='sum',
                                                          activation_fn_in_separable_conv=False,
                                                          stride=1)
        self.middle_flow_block1_unit_11 = xception_module(inplanes=728,
                                                          depth_list=[728, 728, 728],
                                                          skip_connection_type='sum',
                                                          activation_fn_in_separable_conv=False,
                                                          stride=1)
        self.middle_flow_block1_unit_12 = xception_module(inplanes=728,
                                                          depth_list=[728, 728, 728],
                                                          skip_connection_type='sum',
                                                          activation_fn_in_separable_conv=False,
                                                          stride=1)
        self.middle_flow_block1_unit_13 = xception_module(inplanes=728,
                                                          depth_list=[728, 728, 728],
                                                          skip_connection_type='sum',
                                                          activation_fn_in_separable_conv=False,
                                                          stride=1)
        self.middle_flow_block1_unit_14 = xception_module(inplanes=728,
                                                          depth_list=[728, 728, 728],
                                                          skip_connection_type='sum',
                                                          activation_fn_in_separable_conv=False,
                                                          stride=1)
        self.middle_flow_block1_unit_15 = xception_module(inplanes=728,
                                                          depth_list=[728, 728, 728],
                                                          skip_connection_type='sum',
                                                          activation_fn_in_separable_conv=False,
                                                          stride=1)
        self.middle_flow_block1_unit_16 = xception_module(inplanes=728,
                                                          depth_list=[728, 728, 728],
                                                          skip_connection_type='sum',
                                                          activation_fn_in_separable_conv=False,
                                                          stride=1)

        # Exit flow
        self.exit_flow_block1_unit_1 = xception_module(inplanes=728,
                                                       depth_list=[728, 1024, 1024],
                                                       skip_connection_type='conv',
                                                       activation_fn_in_separable_conv=False,
                                                       stride=1)
        self.exit_flow_block2_unit_1 = xception_module(inplanes=1024,
                                                       depth_list=[1536, 1536, 2048],
                                                       skip_connection_type='none',
                                                       activation_fn_in_separable_conv=True,
                                                       stride=1,
                                                       dilation=2)

        self._init_weight()

    def forward(self, x):
        # Entry flow

        x = self.entry_flow_conv1_1(x)
        x = self.entry_flow_bn1_1(x)
        x = self.entry_flow_relu1_1(x)
        x = self.entry_flow_conv1_2(x)
        x = self.entry_flow_bn1_2(x)
        x = self.entry_flow_relu1_2(x)

        x = self.entry_flow_block1_unit_1(x)
        x, low_level_feat = self.entry_flow_block2_unit_1(x)
        x = self.entry_flow_block3_unit_1(x)

        # Middle flow
        x = self.middle_flow_block1_unit_1(x)
        x = self.middle_flow_block1_unit_2(x)
        x = self.middle_flow_block1_unit_3(x)
        x = self.middle_flow_block1_unit_4(x)
        x = self.middle_flow_block1_unit_5(x)
        x = self.middle_flow_block1_unit_6(x)
        x = self.middle_flow_block1_unit_7(x)
        x = self.middle_flow_block1_unit_8(x)
        x = self.middle_flow_block1_unit_9(x)
        x = self.middle_flow_block1_unit_10(x)
        x = self.middle_flow_block1_unit_11(x)
        x = self.middle_flow_block1_unit_12(x)
        x = self.middle_flow_block1_unit_13(x)
        x = self.middle_flow_block1_unit_14(x)
        x = self.middle_flow_block1_unit_15(x)
        x = self.middle_flow_block1_unit_16(x)

        # Exit flow
        x = self.exit_flow_block1_unit_1(x)
        x = self.exit_flow_block2_unit_1(x)

        return x, low_level_feat

    def _init_weight(self):
        self.entry_flow_bn1_1.weight.data.fill_(1)
        self.entry_flow_bn1_1.bias.data.zero_()
        self.entry_flow_bn1_2.weight.data.fill_(1)
        self.entry_flow_bn1_2.bias.data.zero_()


class xception_module(nn.Module):
    def __init__(self,
                 inplanes,
                 depth_list,
                 skip_connection_type,
                 stride,
                 unit_rate_list=None,
                 dilation=1,
                 activation_fn_in_separable_conv=True,
                 low_level_features=False):
        super(xception_module, self).__init__()

        if len(depth_list) != 3:
            raise ValueError('Expect three elements in depth_list.')

        if unit_rate_list:
            if len(unit_rate_list) != 3:
                raise ValueError('Expect three elements in unit_rate_list.')
        else:
            unit_rate_list = [1, 1, 1]

        residual = inplanes
        self.separable_conv1 = SeparableConv2d_same(residual,
                                                    depth_list[0],
                                                    kernel_size=3,
                                                    stride=1,
                                                    dilation=dilation * unit_rate_list[0],
                                                    activation_fn_in_separable_conv=activation_fn_in_separable_conv)
        residual = depth_list[0]
        self.separable_conv2 = SeparableConv2d_same(residual,
                                                    depth_list[1],
                                                    kernel_size=3,
                                                    stride=1,
                                                    dilation=dilation * unit_rate_list[1],
                                                    activation_fn_in_separable_conv=activation_fn_in_separable_conv)
        residual = depth_list[1]
        self.separable_conv3 = SeparableConv2d_same(residual,
                                                    depth_list[2],
                                                    kernel_size=3,
                                                    stride=stride,
                                                    dilation=dilation * unit_rate_list[2],
                                                    activation_fn_in_separable_conv=activation_fn_in_separable_conv)

        shortcut_list = []
        if skip_connection_type == 'conv':
            shortcut_list.append(nn.Conv2d(inplanes, depth_list[-1], kernel_size=1, stride=stride, bias=False))
            shortcut_list.append(nn.SyncBatchNorm(depth_list[-1], eps=1e-03, momentum=0.0003))
        self.shortcut = nn.Sequential(*shortcut_list)
        self.skip_connection_type = skip_connection_type
        self.low_level_features = low_level_features

        self._init_weight()

    def forward(self, x):
        x1 = self.separable_conv1(x)

        x2 = self.separable_conv2(x1)

        x3 = self.separable_conv3(x2)
        x4 = self.shortcut(x)
        if self.skip_connection_type == 'conv':
            y = x3 + x4
        elif self.skip_connection_type == 'sum':
            y = x3 + x
        elif self.skip_connection_type == 'none':
            y = x3
        else:
            raise ValueError('Unsupported skip connection type.')

        if self.low_level_features:
            return y, x2
        else:
            return y

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, std=0.09)
            elif isinstance(m, nn.SyncBatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class resnet_utils_conv2d_same(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(resnet_utils_conv2d_same, self).__init__()

        self.stride = stride
        if stride == 1:
            self.conv = nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=1, dilation=dilation, bias=bias)
        else:
            self.conv = nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=0, dilation=dilation, bias=bias)

        self._init_weight()

    def forward(self, x):
        if self.stride != 1:
            x = fixed_padding(x, self.conv.kernel_size[0], rate=self.conv.dilation[0])
        x = self.conv(x)
        return x

    def _init_weight(self):
        n = self.conv.kernel_size[0] * self.conv.kernel_size[1] * self.conv.out_channels
        self.conv.weight.data.normal_(0, math.sqrt(2. / n))


class SeparableConv2d_same(nn.Module):
    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 bias=False,
                 activation_fn_in_separable_conv=True):
        super(SeparableConv2d_same, self).__init__()
        self.relu = nn.ReLU(inplace=False)
        self.activation_fn_in_separable_conv = activation_fn_in_separable_conv
        self.depthwise = nn.Conv2d(inplanes, inplanes, kernel_size, stride, 0, dilation, groups=inplanes, bias=bias)
        self.depthwise_bn = nn.SyncBatchNorm(inplanes, eps=1e-03, momentum=0.0003)
        if activation_fn_in_separable_conv:
            self.depthwise_relu = nn.ReLU(inplace=True)

        self.pointwise = nn.Conv2d(inplanes,
                                   planes,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0,
                                   dilation=1,
                                   groups=1,
                                   bias=bias)
        self.pointwise_bn = nn.SyncBatchNorm(planes, eps=1e-03, momentum=0.0003)
        if activation_fn_in_separable_conv:
            self.pointwise_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if not self.activation_fn_in_separable_conv:
            x = self.relu(x)
            x = fixed_padding(x, self.depthwise.kernel_size[0], rate=self.depthwise.dilation[0])
            x = self.depthwise(x)
            x = self.depthwise_bn(x)
            x = self.pointwise(x)
            x = self.pointwise_bn(x)

        else:
            x = fixed_padding(x, self.depthwise.kernel_size[0], rate=self.depthwise.dilation[0])
            x = self.depthwise(x)
            x = self.depthwise_bn(x)
            x = self.depthwise_relu(x)
            x = self.pointwise(x)
            x = self.pointwise_bn(x)
            x = self.pointwise_relu(x)
        return x


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


def fixed_padding(inputs, kernel_size, rate):
    kernel_size_effective = kernel_size + (kernel_size-1) * (rate-1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=dilation,
                     groups=groups,
                     bias=False,
                     dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


if __name__ == "__main__":
    model = deeplabv3plus(n_classes=19)
    model.cuda().eval()

    data = torch.randn(1, 3, 1024, 2048).cuda()
    with torch.no_grad():
        out = model(data)
        print(out.shape)
