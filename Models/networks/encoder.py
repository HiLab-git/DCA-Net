import math
import torch
import torch.nn as nn

class EncoderDC(nn.Module):
    def __init__(self, BatchNorm):
        super(EncoderDC, self).__init__()
        # if backbone == 'drn':
        #     inplanes = 512
        # elif backbone == 'mobilenet':
        #     inplanes = 320
        # else:
        #     inplanes = 2048
        inplanes = 256
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn = BatchNorm(inplanes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x_maxpool = self.max_pool(x)
        x_avgpool = self.avg_pool(x)

        x_maxpool = self.bn(x_maxpool)
        x_maxpool = self.relu(x_maxpool)
        x_avgpool = self.bn(x_avgpool)
        x_avgpool = self.relu(x_avgpool)

        return x_maxpool, x_avgpool

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            # elif isinstance(m, SynchronizedBatchNorm2d):
            #     m.weight.data.fill_(1)
            #     m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_encoderDC(BatchNorm):
    return EncoderDC(BatchNorm)
