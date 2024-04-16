import torch
import numpy as np
import torch.nn as nn


# # SE block add to U-net
def conv3x3(in_planes, out_planes, stride=1, bias=False, group=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, groups=group, bias=bias)


class SE_Conv_Block(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_out=False):
        super(SE_Conv_Block, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes * 2)
        self.bn2 = nn.BatchNorm2d(planes * 2)
        self.conv3 = conv3x3(planes * 2, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dropout = drop_out

        if planes <= 16:
            self.globalAvgPool = nn.AvgPool2d((224, 300), stride=1)  # (224, 300) for ISIC2018
            self.globalMaxPool = nn.MaxPool2d((224, 300), stride=1)
        elif planes == 32:
            self.globalAvgPool = nn.AvgPool2d((112, 150), stride=1)  # (112, 150) for ISIC2018
            self.globalMaxPool = nn.MaxPool2d((112, 150), stride=1)
        elif planes == 64:
            self.globalAvgPool = nn.AvgPool2d((56, 75), stride=1)    # (56, 75) for ISIC2018
            self.globalMaxPool = nn.MaxPool2d((56, 75), stride=1)
        elif planes == 128:
            self.globalAvgPool = nn.AvgPool2d((28, 37), stride=1)    # (28, 37) for ISIC2018
            self.globalMaxPool = nn.MaxPool2d((28, 37), stride=1)
        elif planes == 256:
            self.globalAvgPool = nn.AvgPool2d((14, 18), stride=1)    # (14, 18) for ISIC2018
            self.globalMaxPool = nn.MaxPool2d((14, 18), stride=1)

        self.fc1 = nn.Linear(in_features=planes * 2, out_features=round(planes / 2))
        self.fc2 = nn.Linear(in_features=round(planes / 2), out_features=planes * 2)
        self.sigmoid = nn.Sigmoid()

        self.downchannel = None
        if inplanes != planes:
            self.downchannel = nn.Sequential(nn.Conv2d(inplanes, planes * 2, kernel_size=1, stride=stride, bias=False),
                                             nn.BatchNorm2d(planes * 2),)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downchannel is not None:
            residual = self.downchannel(x)

        original_out = out
        out1 = out
        # For global average pool
        out = self.globalAvgPool(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.view(out.size(0), out.size(1), 1, 1)
        avg_att = out
        out = out * original_out
        # For global maximum pool
        out1 = self.globalMaxPool(out1)
        out1 = out1.view(out1.size(0), -1)
        out1 = self.fc1(out1)
        out1 = self.relu(out1)
        out1 = self.fc2(out1)
        out1 = self.sigmoid(out1)
        out1 = out1.view(out1.size(0), out1.size(1), 1, 1)
        max_att = out1
        out1 = out1 * original_out

        att_weight = avg_att + max_att
        out += out1
        out += residual
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        if self.dropout:
            out = nn.Dropout2d(0.5)(out)

        return out, att_weight


# Domain SE block to U-net
class Domain_SE_Conv_Block(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, num_adaptor=4, downsample=None, drop_out=False):
        super(Domain_SE_Conv_Block, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dropout = drop_out

        self.domain_adaptor = Adapt_SE_block(num_block=num_adaptor, planes=planes, shrink=2)

        self.downchannel = None
        if inplanes != planes:
            self.downchannel = nn.Sequential(nn.Conv2d(inplanes, planes * 2, kernel_size=1, stride=stride, bias=False),
                                             nn.BatchNorm2d(planes * 2),)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.domain_adaptor(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        if self.dropout:
            out = nn.Dropout2d(0.5)(out)

        return out


# This module is for Domain_SE_Conv_Block
class Adapt_SE_block(nn.Module):
    def __init__(self, num_block, planes, shrink):
        super(Adapt_SE_block, self).__init__()
        self.conv1 = nn.Conv2d(planes, num_block, kernel_size=1, stride=1, padding=0)
        if planes == 128:
            self.size = 48
        if planes == 64:
            self.size = 96
        if planes == 32:
            self.size = 192
        if planes == 16:
            self.size = 384

        self.glb_avg = nn.AvgPool2d(kernel_size=self.size, stride=1)
        self.glb_max = nn.MaxPool2d(kernel_size=self.size, stride=1)

        self.num_adaptor = []
        for i in range(num_block):

            self.num_adaptor.append(nn.Sequential(
                nn.Linear(in_features=planes, out_features=planes // shrink),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=planes // shrink, out_features=planes),
                nn.Sigmoid()
            ))
        # The channel number may be error
        self.fc1 = nn.Linear(in_features=num_block, out_features=round(num_block / 2))
        self.fc2 = nn.Linear(in_features=round(num_block / 2), out_features=num_block)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        residual = x
        domain_x = self.conv1(x)

        # For global average pool
        out_avg = self.glb_avg(x)
        out_avg = out_avg.view(x.size(0), -1)
        avg_recal_feature = []
        max_recal_feature = []
        for adaptor in self.num_adaptor:
            adaptor.cuda()
            out_avg_weight = adaptor(out_avg)
            out_avg_weight = out_avg_weight.unsqueeze(dim=1)
            avg_recal_feature.append(out_avg_weight)
        avg_recal_feature_att = avg_recal_feature[0]
        for k in range(1, len(avg_recal_feature)):
            avg_recal_feature_att = torch.cat((avg_recal_feature_att, avg_recal_feature[k]), dim=1)
        avg_recal_feature_att.cuda()
        # print(avg_recal_feature_arr.shape)

        # For global maximum pool
        out_max = self.glb_max(x)
        out_max = out_max.view(x.size(0), -1)
        for adaptor in self.num_adaptor:
            out_max_weight = adaptor(out_max)
            out_max_weight = out_max_weight.unsqueeze(dim=1)
            max_recal_feature.append(out_max_weight)
        max_recal_feature_att = max_recal_feature[0]
        for k in range(1, len(max_recal_feature)):
            max_recal_feature_att = torch.cat((max_recal_feature_att, max_recal_feature[k]), dim=1)
        max_recal_feature_att.cuda()
        recal_feature_att = max_recal_feature_att + avg_recal_feature_att
        # print(recal_feature_arr.shape)

        # for domain attention
        # for average pool
        domain_x_avg = self.glb_avg(domain_x)
        domain_x_avg = domain_x_avg.view(domain_x.size(0), -1)
        domain_att_avg = self.fc1(domain_x_avg)
        domain_att_avg = self.relu(domain_att_avg)
        domain_att_avg = self.fc2(domain_att_avg)
        domain_att_avg = self.softmax(domain_att_avg)
        domain_att_avg = domain_att_avg.unsqueeze(dim=2)

        # for max pool
        domain_x_max = self.glb_max(domain_x)
        domain_x_max = domain_x_max.view(domain_x.size(0), -1)
        domain_att_max = self.fc1(domain_x_max)
        domain_att_max = self.relu(domain_att_max)
        domain_att_max = self.fc2(domain_att_max)
        domain_att_max = self.softmax(domain_att_max)
        domain_att_max = domain_att_max.unsqueeze(dim=2)

        recal_feature_att = recal_feature_att.permute(0, 2, 1)
        domain_recal_att = torch.matmul(recal_feature_att, domain_att_avg) + \
                           torch.matmul(recal_feature_att, domain_att_max)
        domain_recal_att = domain_recal_att.unsqueeze(dim=3)
        domain_recal_att = self.sigmoid(domain_recal_att)
        domain_recal_feature = x * domain_recal_att + residual
        # print(domain_recal_feature.shape)

        return domain_recal_feature


# Generating Domain attention block to U-net
class Gen_Domain_Atten_Block(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, num_adaptor=4, downsample=None, drop_out=False):
        super(Gen_Domain_Atten_Block, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dropout = drop_out

        self.domain_adaptor = Gen_Adapt_SE_block(num_block=num_adaptor, planes=planes, shrink=2)

        self.downchannel = None
        if inplanes != planes:
            self.downchannel = nn.Sequential(nn.Conv2d(inplanes, planes * 2, kernel_size=1, stride=stride, bias=False),
                                             nn.BatchNorm2d(planes * 2),)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.domain_adaptor(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        if self.dropout:
            out = nn.Dropout2d(0.5)(out)

        return out


# This module is for Gen_Domain_Atten_Block
class Gen_Adapt_SE_block(nn.Module):
    def __init__(self, num_block, planes, shrink):
        super(Gen_Adapt_SE_block, self).__init__()
        self.num_block = num_block

        self.conv1 = nn.Conv2d(planes, self.num_block, kernel_size=1, stride=1, padding=0)
        if planes == 128:
            self.size = 48
        if planes == 64:
            self.size = 96
        if planes == 32:
            self.size = 192
        if planes == 16:
            self.size = 384

        self.glb_avg = nn.AvgPool2d(kernel_size=self.size, stride=1)
        self.glb_max = nn.MaxPool2d(kernel_size=self.size, stride=1)

        self.domain_adaptor1 = nn.Conv2d(planes*self.num_block, (planes//shrink)*self.num_block, kernel_size=1, stride=1,
                                         padding=0, groups=self.num_block)
        self.domain_norm1 = nn.GroupNorm(num_groups=self.num_block, num_channels=(planes//shrink)*self.num_block)
        self.domain_adaptor2 = nn.Conv2d((planes//shrink)*self.num_block, planes*self.num_block, kernel_size=1, stride=1,
                                         padding=0, groups=self.num_block)
        self.domain_norm2 = nn.GroupNorm(num_groups=self.num_block, num_channels=planes*self.num_block)

        # The channel number may be error
        self.fc1 = nn.Linear(in_features=self.num_block, out_features=round(self.num_block / 2))
        self.fc2 = nn.Linear(in_features=round(self.num_block / 2), out_features=self.num_block)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        residual = x
        domain_x = self.conv1(x)

        # For global average pool
        out_avg = self.glb_avg(x)
        out_avg = out_avg.repeat(1, self.num_block, 1, 1)

        out_avg_weight = self.domain_adaptor1(out_avg)
        out_avg_weight = self.domain_norm1(out_avg_weight)
        out_avg_weight = self.relu(out_avg_weight)
        out_avg_weight = self.domain_adaptor2(out_avg_weight)
        out_avg_weight = self.domain_norm2(out_avg_weight)
        avg_recal_feature_att = self.sigmoid(out_avg_weight)

        # For global maximum pool
        out_max = self.glb_max(x)
        out_max = out_max.repeat(1, self.num_block, 1, 1)
        out_max_weight = self.domain_adaptor1(out_max)
        out_max_weight = self.domain_norm1(out_max_weight)
        out_max_weight = self.relu(out_max_weight)
        out_max_weight = self.domain_adaptor2(out_max_weight)
        out_max_weight = self.domain_norm2(out_max_weight)
        max_recal_feature_att = self.sigmoid(out_max_weight)

        recal_feature_att = max_recal_feature_att + avg_recal_feature_att
        recal_feature_att = recal_feature_att.reshape(recal_feature_att.size(0), self.num_block,
                                                      recal_feature_att.size(1)//self.num_block)
        # print(recal_feature_att.shape)

        # for domain attention
        # for average pool
        domain_x_avg = self.glb_avg(domain_x)
        domain_x_avg = domain_x_avg.view(domain_x.size(0), -1)
        domain_att_avg = self.fc1(domain_x_avg)
        domain_att_avg = self.relu(domain_att_avg)
        domain_att_avg = self.fc2(domain_att_avg)
        domain_att_avg = self.softmax(domain_att_avg)
        domain_att_avg = domain_att_avg.unsqueeze(dim=2)

        # for max pool
        domain_x_max = self.glb_max(domain_x)
        domain_x_max = domain_x_max.view(domain_x.size(0), -1)
        domain_att_max = self.fc1(domain_x_max)
        domain_att_max = self.relu(domain_att_max)
        domain_att_max = self.fc2(domain_att_max)
        domain_att_max = self.softmax(domain_att_max)
        domain_att_max = domain_att_max.unsqueeze(dim=2)
        # print(domain_att_max.size())

        recal_feature_att = recal_feature_att.permute(0, 2, 1)
        domain_recal_att = torch.matmul(recal_feature_att, domain_att_avg) + \
                           torch.matmul(recal_feature_att, domain_att_max)
        domain_recal_att = domain_recal_att.unsqueeze(dim=3)
        domain_recal_att = self.sigmoid(domain_recal_att)
        domain_recal_feature = x * domain_recal_att + residual
        # print(domain_recal_feature.shape)

        return domain_recal_feature
