#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/7/23 8:48 下午
# @Author  : Ran.Gu
'''
Define a dataset class for multi dataset in multi-site task
'''
from torch.utils.data import Dataset
import os, logging, torch
import numpy as np
import random
import SimpleITK as sitk


class MultiDataset_3d(Dataset):
    def __init__(self, nii_dir='./data/clipped_multi_site_data', datalist_dir='./Datasets/Mul_s',
                 ignore_site=None, mode='train'):
        '''
        :param npz_dir:  数据存储位置 './data/npz_data/'
        :param datalist_dir: list存储位置 './Datasets/Mul_s'
        :param mode:模式 在 train, valid 和 test 中
        '''
        self.nii_dir = nii_dir
        self.datalist_dir = datalist_dir
        self.mode = mode
        # self.slices = [file for file in os.listdir(self.npz_dir) if not file.startswith('.')]
        # self.slices.sort()

        if self.mode == 'train':
            with open(os.path.join(self.datalist_dir, self.mode + '_data_list.list'),
                      'r') as f:
                self.image_list = f.readlines()
                if ignore_site != 'None':
                    self.image_list = [item.replace('\n', '').split('_')[0] for item in self.image_list if
                                       ignore_site not in item]
                else:
                    self.image_list = [item.replace('\n', '').split('_')[0] for item in self.image_list]
                self.image_list = list(set(self.image_list))
                random.shuffle(self.image_list)
                self.image_list = [os.path.join(self.nii_dir, x) for x in self.image_list]
        elif self.mode in ['val', 'test']:
            with open(os.path.join(self.datalist_dir, self.mode + '_data_list.list'),
                      'r') as f:
                self.image_list = f.readlines()
                if ignore_site:  # 在测试过程中如果写明unseen_site,则默认测unseen_site
                    self.image_list = [item.replace('\n', '').split('_S')[0] for item in self.image_list
                                       if ignore_site not in item]
                self.image_list = list(set(self.image_list))
                self.image_list = [os.path.join(self.nii_dir, x) for x in self.image_list]
        logging.info(f'Creating {self.mode} dataset in {self.nii_dir} with {len(self.image_list)} slices in total.')

    def __len__(self):
        return len(self.image_list)

    def _translate(self, img, shift=10, roll=True):
        direction = ['right', 'left', 'down', 'up']
        i = random.randint(0, 3)
        img = img.copy()
        if direction[i] == 'right':
            right_slice = img[:, -shift:].copy()
            img[:, shift:] = img[:, :-shift]
            if roll:
                img[:, :shift] = np.fliplr(right_slice)
        if direction[i] == 'left':
            left_slice = img[:, :shift].copy()
            img[:, :-shift] = img[:, shift:]
            if roll:
                img[:, -shift:] = left_slice
        if direction[i] == 'down':
            down_slice = img[-shift:, :].copy()
            img[shift:, :] = img[:-shift, :]
            if roll:
                img[:shift, :] = down_slice
        if direction[i] == 'up':
            upper_slice = img[:shift, :].copy()
            img[:-shift, :] = img[shift:, :]
            if roll:
                img[-shift:, :] = upper_slice
        return img

    def _data_augmentation(self, img, gt):
        if random.randint(0, 1) == 1:
            img = img[::-1, ...]
            gt = gt[::-1, ...]
        # if random.randint(0,1) == 1:
        #     shift_pixel = random.randint(0,10)
        #     img = self._translate(img, shift=shift_pixel)
        #     gt = self._translate(gt, shift=shift_pixel)
        return img, gt

    def __getitem__(self, item):
        # slice_name = os.path.join(self.npz_dir, self.slice[item])
        data = sitk.ReadImage(self.image_list[item]+'.nii.gz')
        img = sitk.GetArrayFromImage(data)
        data_gt = sitk.ReadImage(self.image_list[item]+'_segmentation.nii.gz')
        gt = sitk.GetArrayFromImage(data_gt)
        binary_mask = np.ones(gt.shape)
        mean = np.sum(img * binary_mask) / np.sum(binary_mask)
        std = np.sqrt(np.sum(np.square(img - mean) * binary_mask) / np.sum(binary_mask))
        img = (img - mean) / std
        gt[gt == 2] = 1

        if self.mode == 'train':
            img, gt = self._data_augmentation(img, gt)

        # 为pytorch转化维度和格式
        img, gt = img.astype(np.float32), gt.astype(np.float32)

        return {'slice_name': self.image_list[item], 'img': torch.from_numpy(img), 'gt': torch.from_numpy(gt)}


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    np.random.seed(123)  # Numpy module.
    random.seed(123)  # Python random module.
    torch.manual_seed(123)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_default_tensor_type('torch.FloatTensor')
    import matplotlib.pyplot as plt
    dataset = MultiDataset_3d(nii_dir='./data/npz_data', datalist_dir='./Datasets/Mul_s', mode='train')
    # loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
    # temp = 1
    # for batch in loader:
    #     print(temp)
    #     img, gt = batch['img'], batch['gt']
    #     temp += 1
