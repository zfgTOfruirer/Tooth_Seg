# zfg 改进PraNet(backbone+PD)息肉分割中的模块 + （BFEM + FCFM）（增加的针对牙齿分割的“特征交叉融合”与“边界特征提取”） == BFFNet
# zfg python -m lib.PraNet_Res2Net 查看模型

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.ResNet_ALL import res2net50_v1b_26w_4s


# zfg 名为 BasicConv2d 的类，它继承了 nn.Module 类。这个类的目的是定义一个基本的卷积神经网络层，包括卷积、归一化和激活操作。
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


# zfg  AFF 的类，它继承了 nn.Module 类。这个类的目的是定义一个多特征融合（AFF）模块，用于在不同特征之间进行融合。
class AFF(nn.Module):
    '''
    多特征融合 AFF
    '''

    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x, residual):
        xa = x + residual
        xa = self.conv3x3(xa)
        xg = xa
        xl = self.local_att(xa)
        xl = xa * xl + xa
        xlg = xl + xg
        xlg = self.conv3x3(xlg)
        return xlg


# zfg  RFB_modified 的类，它继承了 nn.Module 类。这个类的目的是定义一个名为 RFB 的卷积神经网络模块，该模块包含四个分支，用于提取不同尺度的特征。
class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


# zfg  aggregation 的类，它继承了 nn.Module 类。这个类的目的是定义一个特征聚合模块，用于将不同尺度的特征进行聚合。
class aggregation(nn.Module):

    def __init__(self, channel):
        super(aggregation, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3 * channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x


# zfg 进行“边界特征提取”与“特征交叉融合”，并最终生成多层特征图。
class BFFNet(nn.Module):
    # res2net based encoder decoder
    def __init__(self, channel=32):
        super(BFFNet, self).__init__()
        # ---- ResNet Backbone ----
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        # ---- Receptive Field Block like module ----
        self.rfb2_1 = RFB_modified(512, channel)
        self.rfb3_1 = RFB_modified(1024, channel)
        self.rfb4_1 = RFB_modified(2048, channel)
        # ---- Partial Decoder ----
        self.agg1 = aggregation(channel)
        # ---- reverse attention branch 4 ----
        self.ra4_conv1 = BasicConv2d(2048, 256, kernel_size=1)
        self.ra4_conv2 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv3 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv4 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv5 = BasicConv2d(256, 1, kernel_size=1)
        # ---- reverse attention branch 3 ----
        self.ra3_conv1 = BasicConv2d(1024, 64, kernel_size=1)
        self.ra3_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra3_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra3_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)
        # ---- reverse attention branch 2 ----
        self.ra2_conv1 = BasicConv2d(512, 64, kernel_size=1)
        self.ra2_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)

    def aff(self, channels, i, img1, img2):
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"
        device = torch.device("cuda:0")
        residual0 = img1.expand(-1, channels, -1, -1).to(device)
        model = AFF(channels=channels)
        model = model.to(device).train()
        x = model(img2, residual0)  # zfg  x0, crop_4
        if i == 2:
            x = self.ra2_conv1(x)
            x = F.relu(self.ra2_conv2(x))
            x = F.relu(self.ra2_conv3(x))
            x = self.ra2_conv4(x)

        if i == 3:
            x = self.ra3_conv1(x)
            x = F.relu(self.ra3_conv2(x))
            x = F.relu(self.ra3_conv3(x))
            x = self.ra3_conv4(x)

        if i == 4:
            x = self.ra4_conv1(x)
            x = F.relu(self.ra4_conv2(x))
            x = F.relu(self.ra4_conv3(x))
            x = F.relu(self.ra4_conv4(x))
            x = self.ra4_conv5(x)

        return x

    def forward(self, x):
        # zfg 卷积特征提取层
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)  # bs, 64, 88, 88
        # ---- low-level features ----
        # zfg 第一个卷积
        x1 = self.resnet.layer1(x)  # bs, 256, 88, 88
        x2 = self.resnet.layer2(x1)  # bs, 512, 44, 44
        # zfg 第二个卷积
        x3 = self.resnet.layer3(x2)  # bs, 1024, 22, 22
        x4 = self.resnet.layer4(x3)  # bs, 2048, 11, 11
        # zfg 第三个则进入平行编码器

        # zfg 三个，每个三层的编码层  三个高级特征
        x2_rfb = self.rfb2_1(x2)  # channel -> 32
        x3_rfb = self.rfb3_1(x3)  # channel -> 32
        x4_rfb = self.rfb4_1(x4)  # channel -> 32
        # zfg 三个，每个三层的编码层————PD层处拼接处理
        ra5_feat = self.agg1(x4_rfb, x3_rfb, x2_rfb)  # zfg 0,1,2,3,4:聚合高级特征2，3，4
        lateral_map_5 = F.interpolate(ra5_feat, scale_factor=8,
                                      mode='bilinear')  # zfg 对应为Global Map（全局映射图）  # NOTES: Sup-1 (bs, 1, 44, 44) -> (bs, 1, 352, 352)

        # zfg  数据输入边界特征提取机制
        # zfg 第一个下采样 (scale_factor=0.25）
        crop_4 = F.interpolate(ra5_feat, scale_factor=0.25, mode='bilinear')  # zfg 下采样后的结果要输入BFEM,+,FCFM(3 条输入)

        # zfg 第一个BFEM
        # ---- reverse attention branch_4 ----
        x = -1 * (torch.sigmoid(crop_4)) + 1  # zfg 激活函数---二值化图像----1表示反转----ROI变为0像素---表示反向注意力的权重
        x = x.expand(-1, 2048, -1, -1).mul(x4)  # zfg 卷积的最后一层（x4）与x(二值化图像)相乘==反向注意力的特征
        x = x + x4  # zfg new
        x = self.ra4_conv1(x)  # zfg 卷积操作

        x = F.relu(self.ra4_conv2(x))  # zfg 对3*3卷积后+激活函数操作
        x = F.relu(self.ra4_conv3(x))
        x = F.relu(self.ra4_conv4(x))

        ra4_feat = self.ra4_conv5(x)  # zfg 卷积操作
        # zfg 第一个+
        x0 = ra4_feat + crop_4
        # zfg 第一个FCFM
        x = self.aff(2048, 4, x0, crop_4)

        lateral_map_4 = F.interpolate(x, scale_factor=32,
                                      mode='bilinear')  # NOTES: Sup-2 (bs, 1, 11, 11) -> (bs, 1, 352, 352)
        # zfg 第二个BFEM
        # ---- reverse attention branch_3 ----
        # zfg 第一个上采样(传入第二个BFEM 与 +)
        crop_3 = F.interpolate(x, scale_factor=2, mode='bilinear')  # zfg 后续RA(先上采样 scale_factor=0.25）

        x = -1 * (torch.sigmoid(crop_3)) + 1
        x = x.expand(-1, 1024, -1, -1).mul(x3)
        x = x + x3

        x = self.ra3_conv1(x)
        x = F.relu(self.ra3_conv2(x))
        x = F.relu(self.ra3_conv3(x))
        ra3_feat = self.ra3_conv4(x)

        # zfg 第二个+
        x1 = ra3_feat + crop_3
        # zfg 第二个FCFM
        x = self.aff(1024, 3, x1, crop_3)

        lateral_map_3 = F.interpolate(x, scale_factor=16,
                                      mode='bilinear')  # NOTES: Sup-3 (bs, 1, 22, 22) -> (bs, 1, 352, 352)
        # zfg 第三个BFEM
        # ---- reverse attention branch_2 ----
        # zfg 第二个上采样(传入第三个BFEM 与 +)
        crop_2 = F.interpolate(x, scale_factor=2, mode='bilinear')

        x = -1 * (torch.sigmoid(crop_2)) + 1
        x = x.expand(-1, 512, -1, -1).mul(x2)
        x = x + x2

        x = self.ra2_conv1(x)
        x = F.relu(self.ra2_conv2(x))
        x = F.relu(self.ra2_conv3(x))
        ra2_feat = self.ra2_conv4(x)

        # zfg 第三个+
        x2 = ra2_feat + crop_2
        # zfg 第三个FCFM
        x = self.aff(512, 2, x2, crop_2)

        lateral_map_2 = F.interpolate(x, scale_factor=8,
                                      mode='bilinear')  # NOTES: Sup-4 (bs, 1, 44, 44) -> (bs, 1, 352, 352)

        return lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2  # zfg 返回多层特征图


if __name__ == '__main__':
    ras = BFFNet().cuda()
    input_tensor = torch.randn(4, 3, 256, 256).cuda()

    out = ras(input_tensor)
    total_params = sum(p.numel() for p in ras.parameters())
    print(f"Total number of parameters: {total_params}")

    print(out)
    print(BFFNet())
