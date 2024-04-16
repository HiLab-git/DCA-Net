import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from Models.layers.modules import conv_block, UpCat, UpCatconv, UnetDsv3


# U-net for medical image segmentation
class forward_Unet(nn.Module):
    def __init__(self, net_param):
        super(forward_Unet, self).__init__()
        self.in_channels = net_param['num_channels']
        self.num_classes = net_param['num_classes']
        self.feature_scale = net_param['feature_scale']

        filters = [64, 128, 256, 512, 1024]    # 64, 128, 256, 512, 1024
        filters = [int(x / self.feature_scale) for x in filters]

        self.conv1 = conv_block(self.in_channels, filters[0])
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = conv_block(filters[0], filters[1])
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = conv_block(filters[1], filters[2])
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = conv_block(filters[2], filters[3], drop_out=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        self.center = conv_block(filters[3], filters[4], drop_out=True)

        # 逆卷积，也可以使用上采样
        self.up4 = UpCatconv(filters[4], filters[3], drop_out=True)
        # self.upconv4 = conv_block(filters[4], filters[3])
        self.up3 = UpCatconv(filters[3], filters[2])
        # self.upconv3 = conv_block(filters[3], filters[2])
        self.up2 = UpCatconv(filters[2], filters[1])
        self.dsv2 = nn.Upsample(size=net_param['output_size'], mode='bilinear')
        # self.upconv2 = conv_block(filters[2], filters[1])
        self.up1 = UpCatconv(filters[1], filters[0])
        # self.upconv1 = conv_block(filters[1], filters[0])

        self.final_conv = nn.Conv2d(filters[0], self.num_classes, kernel_size=1)
        self.softmax = nn.Softmax2d()

    def forward(self, x):
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)
        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)
        center = self.center(pool4)

        up_4 = self.up4(conv4, center)
        up_3 = self.up3(conv3, up_4)
        up_2 = self.up2(conv2, up_3)
        up_2s = self.dsv2(up_2)
        up_1 = self.up1(conv1, up_2)
        logits = self.final_conv(up_1)
        pred_prob = self.softmax(logits)
        pred_compact = torch.argmax(pred_prob, keepdim=True)
        embeddings = torch.cat([up_2s, up_1], dim=1)

        return pred_prob, pred_compact, embeddings


class forward_metric_net(nn.Module):
    def __init__(self):
        super(forward_metric_net, self).__init__()
        self.linear1 = nn.Linear(in_features=48, out_features=24)
        self.linear2 = nn.Linear(in_features=24, out_features=16)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        out = self.linear1(x)
        out = self.leaky_relu(out)
        out = self.linear2(out)
        out = self.leaky_relu(out)

        return out
