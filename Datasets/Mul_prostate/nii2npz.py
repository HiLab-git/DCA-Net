#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/7/23 4:35 下午
# @Author  : Jingyang.Zhang
'''
处理 multi-site 数据
'''
import SimpleITK as sitk
import numpy as np
import os
import scipy.ndimage
import matplotlib.pyplot as plt
import shutil

def _load_normalized_vol(data_path):
    '''
    从路径中读取体数据和分割标签
    :param data_path: 单个病人的体数据和分割标签的 路径
    :return: 体数据,分割和病人名称
    '''
    path = data_path.split(',')
    image_path = path[0]
    label_path = path[1]
    case_name = image_path[-13:-7]
    print('produce %s' % case_name)

    # 读入每个病人的体数据 和 分割标注
    itk_image = sitk.ReadImage(image_path)
    itk_mask = sitk.ReadImage(label_path)
    image = sitk.GetArrayFromImage(itk_image)
    mask = sitk.GetArrayFromImage(itk_mask)
    binary_mask = np.ones(mask.shape)
    mean = np.sum(image * binary_mask) / np.sum(binary_mask)
    std = np.sqrt(np.sum(np.square(image - mean) * binary_mask) / np.sum(binary_mask))
    image = (image - mean) / std
    mask[mask == 2] = 1

    # 旋转方向 和 数据验证
    image, mask = image.transpose([1,2,0]), mask.transpose([1,2,0])
    assert (image.shape[0] == 384) and (image.shape[1] == 384)
    assert (mask.shape[0] == 384) and (mask.shape[1] == 384)
    assert len(np.unique(mask)) == 2
    return image, mask, case_name


def _patch_center_z(mask):
    limX, limY, limZ = np.where(mask > 0)
    z = np.arange(max(1, np.min(limZ)), min(np.max(limZ), mask.shape[2] - 2) + 1)
    return z


def _label_decomp(label_vol, num_cls):
    """
    decompose label for softmax classifier
    original labels are batchsize * W * H * 1, with label values 0,1,2,3...
    this function decompse it to one hot, e.g.: 0,0,0,1,0,0 in channel dimension
    numpy version of tf.one_hot
    """
    one_hot = []
    for i in range(num_cls):
        _vol = np.zeros(label_vol.shape)
        _vol[label_vol == i] = 1
        one_hot.append(_vol)

    return np.stack(one_hot, axis=-1)


def _extract_patch(image, mask):
    '''
    :param image: 单个病人的体数据
    :param mask: 单个病人的分割标签
    :return: 输出数据增强之后的 patches 为了后续训练,
            image_patch-[384, 384, 3]; mask_patch-[384, 384, 2]
    '''
    slice_indexs = []
    image_patches = []
    mask_patches = []

    z = _patch_center_z(mask)
    for z_i in z:
        image_patch = image[:,:,z_i-1:z_i+2]
        mask_patch = mask[:,:,z_i]
        mask_patch = _label_decomp(mask_patch, 2)

        image_patches.append(image_patch)
        mask_patches.append(mask_patch)
        slice_indexs.append(z_i)
    return image_patches, mask_patches, slice_indexs


if __name__ == '__main__':

    datasets = ['BIDMC', 'HK', 'I2CVB' ,'ISBI', 'ISBI_1.5', 'UCL']
    for dataset in datasets:
        datalist = '../dataset/%s_train_list' % dataset
        output_dataset = '../dataset/npz_data/%s' % dataset
        shutil.rmtree(output_dataset, ignore_errors=True)
        os.makedirs(output_dataset, exist_ok=True)

        with open(datalist, 'r') as fp:
            rows = fp.readlines()
        image_list = [row[:-1] for row in rows]

        for data_path in image_list:
            image, mask, case_name = _load_normalized_vol(data_path)  # 单个病人的体数据和分割标注
            image_patches, mask_patches, slice_indexs = _extract_patch(image, mask)

            for i, slice_index in enumerate(slice_indexs):
                slice_name = '%s_Slice%d' % (case_name, slice_index)
                np.savez(os.path.join(output_dataset, slice_name), image_patches[i], mask_patches[i])

