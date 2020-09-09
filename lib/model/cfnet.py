import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class CFNet(nn.Module):
    def __init__(self, in_planes=3, n_classes=19):

        super(CFNet, self).__init__()

        self.conv1 = self.conv(in_planes, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = self.conv(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = self.conv(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3_1 = self.conv(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = self.conv(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv4_1 = self.conv(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = self.conv(256, 256, kernel_size=3, stride=2, padding=1)
        self.conv5_1 = self.conv(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv6 = self.conv(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv6_1 = self.conv(512, 512, kernel_size=3, stride=1, padding=1)

        self.deconv5 = self.deconv(512, 256)
        self.deconv4 = self.deconv(512, 128)
        self.deconv3 = self.deconv(384, 64)
        self.deconv2 = self.deconv(192, 32)

        self.predict_cc = nn.Conv2d(96, n_classes, 1, stride=1, padding=0)

    def forward(self, image):
        out_conv1 = self.conv1(image)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        out_deconv5 = self.antipad(self.deconv5(out_conv6),
                                   evenh=out_conv5.shape[2] % 2 == 0,
                                   evenw=out_conv5.shape[3] % 2 == 0)
        concat5 = torch.cat((out_conv5, out_deconv5), 1)

        out_deconv4 = self.antipad(self.deconv4(concat5),
                                   evenh=out_conv4.shape[2] % 2 == 0,
                                   evenw=out_conv4.shape[3] % 2 == 0)
        concat4 = torch.cat((out_conv4, out_deconv4), 1)

        out_deconv3 = self.antipad(self.deconv3(concat4),
                                   evenh=out_conv3.shape[2] % 2 == 0,
                                   evenw=out_conv3.shape[3] % 2 == 0)
        concat3 = torch.cat((out_conv3, out_deconv3), 1)

        out_deconv2 = self.antipad(self.deconv2(concat3),
                                   evenh=out_conv2.shape[2] % 2 == 0,
                                   evenw=out_conv2.shape[3] % 2 == 0)
        concat2 = torch.cat((out_conv2, out_deconv2), 1)

        correction_cue = self.predict_cc(concat2)

        return correction_cue

    def conv(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.SyncBatchNorm(out_planes), nn.LeakyReLU(0.1, inplace=True))

    def deconv(self, in_planes, out_planes):
        return nn.Sequential(nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=0, bias=False),
                             nn.LeakyReLU(0.1, inplace=True))

    def antipad(self, tensor, evenh=True, evenw=True, num=1):

        h = tensor.shape[2]
        w = tensor.shape[3]
        if evenh and evenw:
            tensor = tensor.narrow(2, 1, h - 2*num)
            tensor = tensor.narrow(3, 1, w - 2*num)
            return tensor
        elif evenh and (not evenw):
            tensor = tensor.narrow(2, 1, h - 2*num)
            tensor = tensor.narrow(3, 1, w - 2*num - 1)
            return tensor
        elif (not evenh) and evenw:
            tensor = tensor.narrow(2, 1, h - 2*num - 1)
            tensor = tensor.narrow(3, 1, w - 2*num)
            return tensor
        else:
            tensor = tensor.narrow(2, 1, h - 2*num - 1)
            tensor = tensor.narrow(3, 1, w - 2*num - 1)
            return tensor


if __name__ == '__main__':
    model = CFNet()
    model.cuda().eval()

    data = torch.rand([1, 3, 1024, 2048]).cuda()

    with torch.no_grad():
        out = model(data)
        print(out.shape)