import torch
import torch.nn as nn
from Models.layers.modules import conv_block, UpCat, UnetDsv3
from Models.layers.channel_attention_layer_g import Gen_Domain_Atten_Block


class Gen_Domain_Atten_Unet(nn.Module):
    def __init__(self, net_param):
        super(Gen_Domain_Atten_Unet, self).__init__()
        self.is_deconv = net_param['deconv']
        self.is_desupe = net_param['desupe']
        self.is_batchnorm = net_param['batchnorm']
        self.in_channels = net_param['num_channels']
        self.num_classes = net_param['num_classes']
        self.feature_scale = net_param['feature_scale']
        self.out_size = net_param['output_size']
        self.num_adaptor = net_param['num_adaptor']

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = conv_block(self.in_channels, filters[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = conv_block(filters[0], filters[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv3 = conv_block(filters[1], filters[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv4 = conv_block(filters[2], filters[3], drop_out=True)
        self.maxpool4 = nn.MaxPool2d(kernel_size=(2, 2))
        self.center = conv_block(filters[3], filters[4], drop_out=True)

        # upsampling
        self.up_concat4 = UpCat(filters[4], filters[3], self.is_deconv)
        self.up4 = Gen_Domain_Atten_Block(filters[4], filters[3], 1, self.num_adaptor, drop_out=True)
        self.up_concat3 = UpCat(filters[3], filters[2], self.is_deconv)
        self.up3 = Gen_Domain_Atten_Block(filters[3], filters[2], 1, self.num_adaptor)
        self.up_concat2 = UpCat(filters[2], filters[1], self.is_deconv)
        self.up2 = Gen_Domain_Atten_Block(filters[2], filters[1], 1, self.num_adaptor)
        self.up_concat1 = UpCat(filters[1], filters[0], self.is_deconv)
        self.up1 = Gen_Domain_Atten_Block(filters[1], filters[0], 1, self.num_adaptor)

        # deep supervision
        if self.is_desupe:
            self.dsv4 = UnetDsv3(in_size=filters[3], out_size=4, scale_factor=self.out_size)
            self.dsv3 = UnetDsv3(in_size=filters[2], out_size=4, scale_factor=self.out_size)
            self.dsv2 = UnetDsv3(in_size=filters[1], out_size=4, scale_factor=self.out_size)
            self.dsv1 = nn.Conv2d(in_channels=filters[0], out_channels=4, kernel_size=1)

        # final conv (without any concat)
        self.final = nn.Sequential(nn.Conv2d(filters[0], filters[0], kernel_size=1),
                                   nn.Conv2d(filters[0], self.num_classes, kernel_size=1))
        self.final1 = nn.Sequential(nn.Conv2d(filters[0], self.num_classes, kernel_size=1), nn.Tanh())
        self.tanh = nn.Tanh()

    def forward(self, inputs, training=True):
        # Feature Extraction
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)
        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)
        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)
        conv4 = self.conv4(maxpool3, training)
        maxpool4 = self.maxpool4(conv4)
        # Center layer
        center = self.center(maxpool4, training)

        dof_dm = []
        # Domain feature recalibrating
        up4 = self.up_concat4(conv4, center)
        print("55555",conv4.shape, center.shape)
        up4, dof4 = self.up4(up4, training)
        # dof_dm4 = [self.tanh(dof_s) for dof_s in dof4]
        # dof_dm.append(dof_dm4)
        up3 = self.up_concat3(conv3, up4)
        up3, dof3 = self.up3(up3)
        # dof_dm3 = [self.tanh(dof_s) for dof_s in dof3]
        # dof_dm.append(dof_dm3)
        up2 = self.up_concat2(conv2, up3)
        up2, dof2 = self.up2(up2)
        # dof_dm2 = [self.tanh(dof_s) for dof_s in dof2]
        # dof_dm.append(dof_dm2)
        up1 = self.up_concat1(conv1, up2)
        up, dof1 = self.up1(up1)
        # dof1_c = self.final1(dof1)
        # dof2_c = self.final1(dof2)
        dof_dm1 = [self.tanh(dof_s) for dof_s in dof1]
        dof_dm.append(dof_dm1)
        # dof2_dm = self.tanh(dof2)

        # Deep Supervision
        if self.is_desupe:
            dsv4 = self.dsv4(up4)
            dsv3 = self.dsv3(up3)
            dsv2 = self.dsv2(up2)
            dsv1 = self.dsv1(up)
            up = torch.cat([dsv1, dsv2, dsv3, dsv4], dim=1)

        seg_out = self.final(up)
        tanh_out = self.final1(up)

        return seg_out, tanh_out, dof_dm
        # return seg_out
