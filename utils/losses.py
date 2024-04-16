#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/5/6 16:55
# @Author  : Ran.Gu
import torch
import numpy as np
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from scipy import ndimage

def get_soft_label(input_tensor, num_class):
    """
        convert a label tensor to soft label
        input_tensor: tensor with shape [N, C, H, W]
        output_tensor: shape [N, H, W, num_class]
    """
    tensor_list = []
    input_tensor = input_tensor.permute(0, 2, 3, 1)
    for i in range(num_class):
        temp_prob = torch.eq(input_tensor, i * torch.ones_like(input_tensor))
        tensor_list.append(temp_prob)
    output_tensor = torch.cat(tensor_list, dim=-1)
    output_tensor = output_tensor.float()
    return output_tensor


class SoftDiceLoss(_Loss):
    '''
    Soft_Dice = 2*|dot(A, B)| / (|dot(A, A)| + |dot(B, B)| + eps)
    eps is a small constant to avoid zero division,
    '''
    def __init__(self, *args, **kwargs):
        super(SoftDiceLoss, self).__init__()

    def forward(self, prediction, soft_ground_truth, num_class=3, weight_map=None, eps=1e-8):
        dice_loss = soft_dice_loss(prediction, soft_ground_truth, num_class, weight_map)

        return dice_loss

def soft_dice_loss(prediction, soft_ground_truth, num_class, weight_map=None):
    predict = prediction.permute(0, 2, 3, 1)
    pred = predict.contiguous().view(-1, num_class)
    # pred = F.softmax(pred, dim=1)
    ground = soft_ground_truth.view(-1, num_class)
    n_voxels = ground.size(0)
    if weight_map is not None:
        weight_map = weight_map.view(-1)
        weight_map_nclass = weight_map.repeat(num_class).view_as(pred)
        ref_vol = torch.sum(weight_map_nclass * ground, 0)
        intersect = torch.sum(weight_map_nclass * ground * pred, 0)
        seg_vol = torch.sum(weight_map_nclass * pred, 0)
    else:
        ref_vol = torch.sum(ground, 0)
        intersect = torch.sum(ground * pred, 0)
        seg_vol = torch.sum(pred, 0)
    dice_score = (2.0 * intersect) / (ref_vol + seg_vol + 1e-5)
    # dice_loss = 1.0 - torch.mean(dice_score)
    # return dice_loss
    dice_score = torch.mean(-torch.log(dice_score))
    return dice_score


class DiceLoss(_Loss):
    '''
    Soft_Dice = 2*|dot(A, B)| / (|dot(A, A)| + |dot(B, B)| + eps)
    eps is a small constant to avoid zero division,
    '''
    def __init__(self, *args, **kwargs):
        super(DiceLoss, self).__init__()

    def forward(self, prediction, soft_ground_truth, num_class=3, weight_map=None, eps=1e-8):
        dice_loss = dice_loss_mute(prediction, soft_ground_truth, num_class, weight_map)

        return dice_loss

def dice_loss_mute(prediction, ground_truth, num_class, weight_map=None):
    predict = prediction.permute(0, 2, 3, 1)
    pred = predict.contiguous().reshape(-1, num_class)
    # pred = F.softmax(pred, dim=1)
    soft_ground = ground_truth.permute(0, 2, 3, 1)
    ground = soft_ground.reshape(-1, num_class)
    n_voxels = ground.size(0)
    if weight_map is not None:
        weight_map = weight_map.view(-1)
        weight_map_nclass = weight_map.repeat(num_class).view_as(pred)
        ref_vol = torch.sum(weight_map_nclass * ground, 0)
        intersect = torch.sum(weight_map_nclass * ground * pred, 0)
        seg_vol = torch.sum(weight_map_nclass * pred, 0)
    else:
        ref_vol = torch.sum(ground, 0)
        intersect = torch.sum(ground * pred, 0)
        seg_vol = torch.sum(pred, 0)
    dice_score = (2.0 * intersect) / (ref_vol + seg_vol + 1e-5)
    dice_loss = 1.0 - torch.mean(dice_score)
    # return dice_loss

    return dice_loss


def val_dice_class(prediction, ground_truth, num_class):
    '''
    calculate the dice loss in each class case in multi-class problem
    '''
    predict = prediction.permute(0, 2, 3, 1)
    pred = predict.contiguous().reshape(-1, num_class)
    # pred = F.softmax(pred, dim=1)
    soft_ground = ground_truth.permute(0, 2, 3, 1)
    ground = soft_ground.reshape(-1, num_class)
    ref_vol = torch.sum(ground, 0)
    intersect = torch.sum(ground * pred, 0)
    seg_vol = torch.sum(pred, 0)
    dice_score = 2.0 * intersect / (ref_vol + seg_vol + 1.0)

    return dice_score


def val_dice(prediction, soft_ground_truth, num_class):
    '''
    calculate the mean dice loss in all case
    '''
    # predict = prediction.permute(0, 2, 3, 1)
    pred = prediction.contiguous().view(-1, num_class)
    # pred = F.softmax(pred, dim=1)
    ground = soft_ground_truth.view(-1, num_class)
    ref_vol = torch.sum(ground, 0)
    intersect = torch.sum(ground * pred, 0)
    seg_vol = torch.sum(pred, 0)
    dice_score = 2.0 * intersect / (ref_vol + seg_vol + 1.0)


def Intersection_over_Union_class(prediction, soft_ground_truth, num_class):
    '''
    calculate the IoU in each class case in multi-class problem
    '''
    # predict = prediction.permute(0, 2, 3, 1)
    pred = prediction.contiguous().view(-1, num_class)
    # pred = F.softmax(pred, dim=1)
    ground = soft_ground_truth.view(-1, num_class)
    ref_vol = torch.sum(ground, 0)
    intersect = torch.sum(ground * pred, 0)
    seg_vol = torch.sum(pred, 0)
    iou_score = intersect / (ref_vol + seg_vol - intersect + 1.0)

    return iou_score


def Intersection_over_Union(prediction, soft_ground_truth, num_class):
    '''
    calculate the mean IoU in all case
    '''
    # predict = prediction.permute(0, 2, 3, 1)
    pred = prediction.contiguous().view(-1, num_class)
    # pred = F.softmax(pred, dim=1)
    ground = soft_ground_truth.view(-1, num_class)
    ref_vol = torch.sum(ground, 0)
    intersect = torch.sum(ground * pred, 0)
    seg_vol = torch.sum(pred, 0)
    iou_score = intersect / (ref_vol + seg_vol - intersect + 1.0)
    iou_mean_score = torch.mean(iou_score)

    return iou_mean_score


def get_compactness_cost(y_pred, y_true):

    """
    y_pred: BxHxWxC
    """
    """
    lenth term
    """

    # y_pred = tf.one_hot(y_pred, depth=2)
    # print (y_true.shape)
    # print (y_pred.shape)
    y_pred = y_pred.permute(0, 2, 3, 1)
    y_pred = y_pred[..., 1]
    y_true = y_pred[..., 1]

    x = y_pred[:, 1:, :] - y_pred[:, :-1, :]    # horizontal and vertical directions
    y = y_pred[:, :, 1:] - y_pred[:, :, :-1]

    delta_x = x[:, :, 1:]**2
    delta_y = y[:, 1:, :]**2

    delta_u = torch.abs(delta_x + delta_y)

    epsilon = 0.00000001 # where is a parameter to avoid square root is zero in practice.
    w = 0.01
    length = w * torch.sum(torch.sqrt(delta_u + epsilon), [1, 2])

    area = torch.sum(y_pred, [1, 2])

    compactness_loss = torch.sum(length ** 2 / (area * 4 * 3.1415926))

    return compactness_loss, torch.sum(length), torch.sum(area), delta_u


def get_coutour_sample(y_true):
    """
    y_true: BxHxWx2
    """
    y_true = y_true.permute(0, 2, 3, 1)
    y_true_arr = y_true.cpu().numpy()
    positive_mask = np.expand_dims(y_true_arr[..., 1], axis=3)
    metrix_label_group = np.expand_dims(np.array([1, 0, 1, 1, 0]), axis=1)
    coutour_group = np.zeros(positive_mask.shape)

    for i in range(positive_mask.shape[0]):
        slice_i = positive_mask[i]

        if metrix_label_group[i] == 1:
            # generate coutour mask
            erosion = ndimage.binary_erosion(slice_i[..., 0], iterations=1).astype(slice_i.dtype)
            sample = np.expand_dims(slice_i[..., 0] - erosion, axis = 2)

        elif metrix_label_group[i] == 0:
            # generate background mask
            dilation = ndimage.binary_dilation(slice_i, iterations=5).astype(slice_i.dtype)
            sample = dilation - slice_i

        coutour_group[i] = sample
    coutour_group = coutour_group.transpose((0, 3, 1, 2))
    return torch.from_numpy(coutour_group).cuda(), torch.from_numpy(metrix_label_group).cuda()


def extract_coutour_embedding(coutour, embeddings):
        coutour_embeddings = coutour * embeddings
        average_embeddings = torch.sum(coutour_embeddings, [2, 3])/torch.sum(coutour, [2, 3])
        # print (coutour.shape)
        # print (embeddings.shape)
        # print (coutour_embeddings.shape)
        # print (average_embeddings.shape)
        return average_embeddings


def connectivity_region_analysis(mask):
    s = [[0, 1, 0],
         [1, 1, 1],
         [0, 1, 0]]
    mask_arr = mask.cpu().numpy()
    label_im, nb_labels = ndimage.label(mask_arr)   #, structure=s)

    sizes = ndimage.sum(mask_arr, label_im, range(nb_labels + 1))

    # plt.imshow(label_im)
    label_im[label_im != np.argmax(sizes)] = 0
    label_im[label_im == np.argmax(sizes)] = 1

    return torch.from_numpy(label_im).cuda().float()


def entropy_loss(p,C=2):
    ## p N*C*W*H*D
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1)/torch.tensor(np.log(C)).cuda()
    ent = torch.mean(y1)

    return ent


def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target)
    z_sum = torch.sum(score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def softmax_dice_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    n = input_logits.shape[1]
    dice = 0
    for i in range(0, n):
        dice += dice_loss(input_softmax[:, i], target_softmax[:, i])
    mean_dice = dice / n

    return mean_dice


def entropy_loss_map(p, C=2):
    ent = -1*torch.sum(p * torch.log(p + 1e-6), dim=1, keepdim=True)/torch.tensor(np.log(C)).cuda()
    return ent


def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax-target_softmax)**2
    return mse_loss


def mse_loss(score, target):
    mse_loss = (score - target) ** 2
    return mse_loss


def kl_loss(score, target):
    kl_div = F.kl_div(score, target, reduction='none')
    return kl_div


def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    # return F.kl_div(input_log_softmax, target_softmax)
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='none')
    # mean_kl_div = torch.mean(0.2*kl_div[:,0,...]+0.8*kl_div[:,1,...])
    return kl_div


def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    return torch.mean((input1 - input2)**2)


def log_gaussian(x, mu, logvar):
    PI = mu.new([np.pi])

    x = x.view(x.shape[0], -1)
    mu = mu.view(x.shape[0], -1)
    logvar = logvar.view(x.shape[0], -1)

    N, D = x.shape

    log_norm = (-1 / 2) * (D * torch.log(2 * PI) +
                           logvar.sum(dim=1) +
                           (((x - mu) ** 2) / (logvar.exp())).sum(dim=1))

    return log_norm