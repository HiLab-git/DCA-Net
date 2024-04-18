import math
import torch
import random
import torch.nn as nn
import torch.nn.functional as F

class MetaEmbedding(nn.Module):
    
    def __init__(self, feat_dim=256, num_domain=3):
        super(MetaEmbedding, self).__init__()
        self.num_domain = num_domain
        self.hallucinator = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(feat_dim, num_domain, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Softmax(1)
        )
        self.selector = nn.Sequential(
            # nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(feat_dim, feat_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh()
        )

    def forward(self, x, domain_code, centroids, *args):
        # storing direct feature
        direct_feature = x
        # hal_scale = self.hallucinator(x)
        hal_scale = torch.softmax(domain_code, -1)

        size = centroids.size()
        centroids_ = centroids.view(centroids.size(0), -1)
        memory_feature = torch.matmul(hal_scale, centroids_)

        memory_feature = memory_feature.view(x.size(0), size[1], size[2], size[3])
        sel_scale = self.selector(x)
        infused_feature = memory_feature * sel_scale
        x = direct_feature + infused_feature
        return x, hal_scale, sel_scale


# This module is for Gen_Domain_Atten_Block
class DomadaptorEmbedding(nn.Module):
    def __init__(self, planes=256, num_block=8, shrink=2):
        super(DomadaptorEmbedding, self).__init__()
        self.num_block = num_block

        self.conv1 = nn.Conv2d(planes, self.num_block, kernel_size=1, stride=1, padding=0)

        self.glb_avg = nn.AdaptiveAvgPool2d((1, 1))
        self.glb_max = nn.AdaptiveMaxPool2d((1, 1))

        self.domain_adaptor1 = nn.Conv2d(planes*self.num_block, (planes*self.num_block//shrink), kernel_size=1, stride=1,
                                         padding=0, groups=self.num_block)
        self.domain_norm1 = nn.GroupNorm(num_groups=self.num_block, num_channels=(planes*self.num_block//shrink))
        self.domain_adaptor2 = nn.Conv2d((planes*self.num_block//shrink), planes*self.num_block, kernel_size=1, stride=1,
                                         padding=0, groups=self.num_block)
        self.domain_norm2 = nn.GroupNorm(num_groups=self.num_block, num_channels=planes*self.num_block)

        # The channel number may be error
        self.fc1 = nn.Linear(in_features=self.num_block, out_features=round(self.num_block / shrink))
        self.fc2 = nn.Linear(in_features=round(self.num_block / shrink), out_features=self.num_block)
        self.relu = nn.ReLU(inplace=True)
        self.leakyrelu = nn.LeakyReLU(inplace=True)
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
        out_avg_weight = self.leakyrelu(out_avg_weight)
        out_avg_weight = self.domain_adaptor2(out_avg_weight)
        out_avg_weight = self.domain_norm2(out_avg_weight)
        avg_recal_feature_att = self.sigmoid(out_avg_weight)

        # For global maximum pool
        out_max = self.glb_max(x)
        out_max = out_max.repeat(1, self.num_block, 1, 1)
        out_max_weight = self.domain_adaptor1(out_max)
        out_max_weight = self.domain_norm1(out_max_weight)
        out_max_weight = self.leakyrelu(out_max_weight)
        out_max_weight = self.domain_adaptor2(out_max_weight)
        out_max_weight = self.domain_norm2(out_max_weight)
        max_recal_feature_att = self.sigmoid(out_max_weight)

        recal_feature_att = max_recal_feature_att + avg_recal_feature_att
        recal_feature_att = recal_feature_att.reshape(recal_feature_att.size(0), self.num_block,
                                                      recal_feature_att.size(1)//self.num_block)

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
        # randomly extract domain feature
        domain_choose_list = random.sample(range(0, self.num_block), 3)
        domain_feature = [x*recal_feature_att[:, :, d:d+1].unsqueeze(dim=3) for d in domain_choose_list]

        # domain dropout
        domain_drop = torch.rand(domain_att_avg.shape[0], domain_att_avg.shape[1]).cuda()
        domain_drop[domain_drop < 0.2] = 0
        domain_drop[domain_drop >= 0.2] = 1
        domain_drop = domain_drop.unsqueeze(dim=2)

        domain_recal_att = torch.matmul(recal_feature_att, domain_att_avg*domain_drop) + \
                           torch.matmul(recal_feature_att, domain_att_max*domain_drop)
        domain_recal_att = domain_recal_att.unsqueeze(dim=3)
        domain_recal_att = self.sigmoid(domain_recal_att)

        domain_recal_feature = x * domain_recal_att + residual

        return domain_recal_feature, domain_recal_att, domain_feature


def build_MetaEmbedding(feat_dim, num_domain):
    return MetaEmbedding(feat_dim, num_domain)
