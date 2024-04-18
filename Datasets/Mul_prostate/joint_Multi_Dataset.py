#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/10/15 12:59 下午
# @Author  : Ran.Gu
'''
Define a dataset class for multi dataset in multi-site task
'''
from torch.utils.data import Dataset
import os, logging, torch
import numpy as np
import random
import SimpleITK as sitk


class joint_Multi_Dataset(Dataset):
    def __init__(self, npz_dir='./data/npz_data', datalist_list=None, mode='train', unseen_site=None, transform=None):
        '''
        :param npz_dir:  数据存储位置 './data/npz_data/'
        :param datalist_dir: list存储位置 './Datasets/Mul_s'
        :param mode:模式 在 train, valid 和 test 中
        '''
        self.npz_dir = npz_dir
        self.mode = mode
        self.datalist_list = datalist_list
        self.image_list = []
        self.transform = transform
        self.unseen_site = unseen_site
        random.seed(2)

        if self.datalist_list is None:
            EOFError('No input data list')

        with open(os.path.join('/media/c1501/f0e0c43d-ac95-418b-8777-9cd59f31e116/lujiangshan/Code/DCA-Net/Datasets/Mul_prostate', str(self.unseen_site) + '_train_list'), 'r') as f:
            data_slice = f.readlines()
            data_slice = [item.replace('\n', '') for item in data_slice]
            data_case = list(set([item.split('/')[-1].split('_')[0] for item in data_slice]))
            valtest_case = random.sample(data_case, round(len(data_case)*0.3))
            valtest_case = [os.path.join(self.unseen_site, case) for case in valtest_case]
        if self.mode == 'train':
            # for datalist in self.datalist_list:
            with open(self.datalist_list, 'r') as f:
                data_path_list = f.readlines()
                data_path_list = [item.replace('\n', '') for item in data_path_list if
                                  item.split('_S')[0] not in valtest_case]
                self.image_list.extend(data_path_list)
            random.shuffle(self.image_list)
        elif self.mode in ['val']:
            with open(self.datalist_list, 'r') as f:
                self.image_list = f.readlines()
                # val_case = random.sample(valtest_case, round(len(valtest_case)/3))
                self.image_list = [item.replace('\n', '') for item in self.image_list if
                                   item.split('_S')[0] in valtest_case]
                self.image_list = random.sample(self.image_list, round(len(self.image_list)*0.3))
        elif self.mode in ['test']:
            with open(self.datalist_list, 'r') as f:
                self.image_list = f.readlines()
                # test_case = random.sample(valtest_case, round(len(valtest_case)*2 / 3))
                self.image_list = [item.replace('\n', '') for item in self.image_list if
                                   item.split('_S')[0] in valtest_case]
        self.image = [os.path.join(self.npz_dir, x + '.npz') for x in self.image_list]
        # logging.info(f'Creating {self.mode} dataset in {[self.npz_dir+"/"+x.split("/")[-1].split("_t")[0] for x in self.datalist_list]} '
        #              f'with {len(self.image)} slices in total.')
        logging.info(
            f'Creating {self.mode} dataset in {self.npz_dir+"/"+self.datalist_list.split("/")[-1].split("_t")[0]} '
            f'with {len(self.image)} slices in total.')

    def __len__(self):
        return len(self.image)

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

        data = np.load(self.image[item])
        img, gt = data['arr_0'], data['arr_1']

        sample = {'slice_name': self.image[item], 'image': img, 'label': gt}
        # if self.mode == 'train':
        #     img, gt = self._data_augmentation(img, gt)
        if self.transform:
            sample = self.transform(sample)

        # 为pytorch转化维度和格式
        # img, gt = img.transpose([2, 0, 1]), gt.transpose([2, 0, 1])
        # img, gt = img.astype(np.float32), gt.astype(np.float32)

        return sample


class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]

        return {'image': image, 'label': label}


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, with_sdf=False):
        self.output_size = output_size
        self.with_sdf = with_sdf

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if self.with_sdf:
            sdf = sample['sdf']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            if self.with_sdf:
                sdf = np.pad(sdf, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape
        # if np.random.uniform() > 0.33:
        #     w1 = np.random.randint((w - self.output_size[0])//4, 3*(w - self.output_size[0])//4)
        #     h1 = np.random.randint((h - self.output_size[1])//4, 3*(h - self.output_size[1])//4)
        # else:
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        if self.with_sdf:
            sdf = sdf[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            return {'image': image, 'label': label, 'sdf': sdf}
        else:
            return {'image': image, 'label': label}


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()

        return {'slice_name': sample['slice_name'], 'image': image, 'label': label}


class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        noise = np.clip(self.sigma * np.random.randn(image.shape[0], image.shape[1], image.shape[2]), -2*self.sigma, 2*self.sigma)
        noise = noise + self.mu
        image = image + noise
        return {'slice_name': sample['slice_name'], 'image': image, 'label': label}


class CreateOnehotLabel(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        onehot_label = np.zeros((self.num_classes, label.shape[0], label.shape[1], label.shape[2]), dtype=np.float32)
        for i in range(self.num_classes):
            onehot_label[i, :, :, :] = (label == i).astype(np.float32)
        return {'slice_name': sample['slice_name'], 'image': image, 'label': label, 'onehot_label': onehot_label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        image, label = image.transpose([2, 0, 1]), label.transpose([2, 0, 1])
        image, label = image.astype(np.float32), label.astype(np.float32)
        # image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        if 'onehot_label' in sample:
            return {'slice_name': sample['slice_name'], 'image': torch.from_numpy(image),
                    'label': torch.from_numpy(label), 'onehot_label': torch.from_numpy(sample['onehot_label'])}
        else:
            return {'slice_name': sample['slice_name'], 'image': torch.from_numpy(image), 'label': torch.from_numpy(label)}


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
    dataset = joint_Multi_Dataset(npz_dir='./data/npz_data', datalist_list='./Datasets/Mul_s', mode='train')
    # loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
    # temp = 1
    # for batch in loader:
    #     print(temp)
    #     img, gt = batch['img'], batch['gt']
    #     temp += 1
