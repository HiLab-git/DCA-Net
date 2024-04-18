#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/10/23 20:11
# @Author  : Ran. Gu
'''
测试使用domain adaptor来解决multi-site的问题，测试是否比直接joint train效果会更好
'''
import os
import torch
import shutil
import random
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from settings import Settings
###
from Datasets.MultiDataset import MultiDataset
from Datasets.Meta_Multi_Dataset import Meta_Multi_Dataset
from utils.dice_loss import get_soft_label, SoftDiceLoss, val_dice, val_dice_class, connectivity_region_analysis
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from Models.networks.unet import Unet
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

### 实验重复性
torch.manual_seed(123)
torch.cuda.manual_seed(123)
torch.cuda.manual_seed_all(123)
np.random.seed(123)  # Numpy module.
random.seed(123)  # Python random module.
torch.manual_seed(123)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.set_default_tensor_type('torch.cuda.FloatTensor')


def _val_on_the_fly(model: nn.Module, loader_val: DataLoader, writer, iter_num):
    '''
    在训练过程中在验证集上验证训练结果，并将其保存到 writer 中
    :param model: 分割模型
    :param loader_val: 验证集 是 list
    :param writer: summarywriter
    :param iter_num: 当前迭代次数
    :return: 验证集上loss
    '''
    model.eval()
    bce_criterion = nn.BCELoss()
    loss_bce_val = 0
    mean_dice_val = 0

    # 依次对每个中心的数据进行预测
    for num, batch_val in enumerate(loader_val):
        img_val, gt_val = batch_val['img'].cuda(), batch_val['gt'].cuda()
        with torch.no_grad():
            pred_val = model(img_val)
        loss_bce_val += bce_criterion(pred_val, gt_val)
        pred_dis = torch.argmax(pred_val, dim=1, keepdim=True)
        pred_dis = connectivity_region_analysis(pred_dis.permute(0, 2, 3, 1))
        pred_dis = pred_dis.permute(0, 3, 1, 2)
        pred_soft = get_soft_label(pred_dis, train_params['num_classes'])  # get soft label
        mean_dice_val += val_dice_class(pred_soft, gt_val, train_params['num_classes'])[1]
        if num % train_params['print_freq'] == 0:
            # summarywriter image
            writer.add_images('val/images', img_val[:, 1:2, :, :], iter_num)
            writer.add_images('val/ground_truths', gt_val[:, 1:2, :, :], iter_num)
            writer.add_images('val/preds', pred_soft.permute(0, 3, 1, 2)[:, 1:2, :, :], iter_num)
    loss_bce_val /= len(loader_val)
    mean_dice_val /= len(loader_val)
    loss_dice_val = 1 - mean_dice_val
    # summarywriter
    writer.add_scalar('val/loss_bce', loss_bce_val, iter_num)
    writer.add_scalar('val/mean_dice', mean_dice_val, iter_num)

    # for tag, value in model.named_parameters():
    #     tag = tag.replace('.', '/')
    #     writer.add_histogram('weights/' + tag, value.cpu().numpy(), iter_num)
    #     writer.add_histogram('grads/' + tag, value.grad.cpu().numpy(), iter_num)

    return loss_bce_val, loss_dice_val, mean_dice_val


def train(model: torch.nn.Module, loader_train: DataLoader, loader_val: DataLoader, train_params: dict, writer):
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=train_params['learning_rate'], weight_decay=train_params['weight_decay'])

    # 定义loss
    bce_criterion = nn.BCELoss()
    dice_crtierion = SoftDiceLoss()

    # 包装迭代器
    loader_iter = iter(loader_train)

    lr_ = train_params['learning_rate']
    best_val_dice = 0
    best_val_step = 0
    for iter_num in tqdm(range(train_params['iterations'])):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader_train)
            batch = next(loader_iter)
            # print(batch['slice_name'])

        # 训练前向传播
        model.train()
        img, gt = batch['img'].cuda(), batch['gt'].cuda()
        pred = model(img)

        # 计算loss
        loss_bce = bce_criterion(pred, gt)
        loss_dice = dice_crtierion(pred, gt, train_params['num_classes'])
        loss = 0.5 * loss_bce + 0.5 * loss_dice

        # 反向传播更新参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # tensorboard信息
        writer.add_scalar('lr', lr_, iter_num)
        writer.add_scalar('loss/loss_bce', loss_bce, iter_num)
        writer.add_scalar('loss/loss_dice', loss_dice, iter_num)
        writer.add_scalar('loss/loss', loss, iter_num)

        # 打印训练过程的loss值
        if iter_num % train_params['print_freq'] == 0:
            logging.info('(Iteration %d, lr: %.4f) --> loss_bce: %.4f; loss_dice: %.4f; loss: %.4f'
                         % (iter_num, lr_, loss_bce.item(), loss_dice.item(), loss.item()))

        # Validation online
        if iter_num % train_params['val_freq'] == 0:
            # calculate the dice
            loss_bce_val, loss_dice_val, mean_dice_val = _val_on_the_fly(model, loader_val, writer, iter_num)
            logging.info('Validation --> loss_bce: %.4f; loss_dice: %.4f; mean_dice: %.4f' %
                         (loss_bce_val.item(), loss_dice_val.item(), mean_dice_val.item()))

            if mean_dice_val > best_val_dice:
                best_val_dice = mean_dice_val
                best_val_step = iter_num
                torch.save(model.state_dict(), os.path.join(train_params['snapshot_path']+data_params['unseen_site'], 'best_model.pth'))
                logging.info('********** Best model (dice: %.4f) is updated at step %d.' %
                             (mean_dice_val.item(), iter_num))
            else:
                logging.info('********** Best model (dice: %.4f) was at step %d, current dice: %.4f.' %
                             (best_val_dice, best_val_step, mean_dice_val.item()))

        # 保存模型
        if iter_num % 6000 == 0:
            save_mode_path = os.path.join(train_params['snapshot_path']+data_params['unseen_site'],
                                          'iter_%d_dice_%.4f.pth' % (iter_num, mean_dice_val.item()))
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        # 改变学习率
        if iter_num % train_params['lr_decay_freq'] == 0:
            lr_ = train_params['learning_rate'] * 0.95 ** (iter_num // train_params['lr_decay_freq'])
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_


if __name__ == '__main__':
    # 配置文件
    settings = Settings()
    common_params, data_params, net_params, train_params, eval_params = settings['COMMON'], settings['DATA'], settings[
        'NETWORK'], settings['TRAINING'], settings['EVAL']

    # 新建任务文件夹
    shutil.rmtree(common_params['exp_dir']+data_params['unseen_site'], ignore_errors=True)
    os.makedirs(common_params['exp_dir']+data_params['unseen_site'], exist_ok=True)
    logging.basicConfig(filename=os.path.join(common_params['exp_dir']+data_params['unseen_site'], 'logs.txt'),
                        level=logging.DEBUG, format='%(asctime)s %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info('Output path = %s' % common_params['exp_dir']+data_params['unseen_site'])

    # 配置训练和测试过程的Dataset
    source_list = ['HK', 'ISBI', 'ISBI_1.5', 'I2CVB', 'UCL', 'BIDMC']
    source_list.remove(data_params['unseen_site'][:-1])
    train_file_list = [os.path.join(data_params['data_list_dir'], source_domain + '_train_list') for source_domain in
                       source_list]
    valid_file_list = [os.path.join(data_params['data_list_dir'], data_params['unseen_site'][:-1] + '_train_list')]
    train_dataset = Meta_Multi_Dataset(npz_dir=data_params['npz_data_dir'], datalist_list=train_file_list, mode='train')
    valid_dataset = Meta_Multi_Dataset(npz_dir=data_params['npz_data_dir'], datalist_list=valid_file_list, mode='val')
    # train_dataset = MultiDataset(npz_dir=data_params['npz_data_dir'], datalist_dir=data_params['data_list_dir'],
    #                              ignore_site=data_params['unseen_site'], mode='train')
    # valid_dataset = MultiDataset(npz_dir=data_params['npz_data_dir'], datalist_dir=data_params['data_list_dir'],
    #                              ignore_site=data_params['unseen_site'], mode='val')
    # test_dataset = MultiDataset(file_dirs=data_params['data_root_dir'], mode='test')  # 用于测试模型
    logging.info('Using MS-site data to do experiment')

    train_loader = DataLoader(dataset=train_dataset, batch_size=train_params['batch_size'],
                              shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=4,
                              shuffle=True, num_workers=0, drop_last=True, pin_memory=True)

    # 配置 summarywriter
    writer = SummaryWriter(log_dir=common_params['exp_dir']+data_params['unseen_site'])

    # 配置分割网络
    model = Unet(net_params).cuda()
    # model = nn.DataParallel(model)

    train(model, train_loader, valid_loader, train_params, writer)
