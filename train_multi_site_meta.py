#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/12/4 20:11
# @Author  : Ran. Gu
'''
大致复现quande liu的工作SAML， Shape-aware Meta-learning for Generalizing Prostate MRI Segmentation to Unseen Domains.
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
import learn2learn as l2l
from settings import Settings
from learn2learn.algorithms import MAML
###
from Datasets.MultiDataset import MultiDataset
from Datasets.Meta_Dataset import Meta_Dataset
from utils.dice_loss import get_soft_label, SoftDiceLoss, val_dice, val_dice_class, get_compactness_cost, get_coutour_sample
from utils.dice_loss import extract_coutour_embedding, connectivity_region_analysis
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from utils.triplet_semihard_loss import TripletSemihardLoss

from Models.networks.meta_unet import forward_Unet, forward_metric_net
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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
            pred_val, pred_com, _ = model(img_val)
        loss_bce_val += bce_criterion(pred_val, gt_val)
        pred_dis = torch.argmax(pred_val, dim=1, keepdim=True)
        pred_dis = connectivity_region_analysis(pred_dis.permute(0, 2, 3, 1))
        pred_dis = pred_dis.permute(0, 3, 1, 2)
        pred_soft = get_soft_label(pred_dis, train_params['num_classes'])  # get soft label
        mean_dice_val += val_dice_class(pred_soft.float(), gt_val, train_params['num_classes'])[1]
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


def train(model: torch.nn.Module, metirc_model: torch.nn.Module, loader_train_list: list, loader_val: DataLoader, train_params: dict, writer):
    # 定义学习率衰减策略
    inner_lr = train_params['inner_lr']
    # 定义模型
    maml_model = MAML(model, inner_lr, first_order=True)

    # 定义优化器
    optimizer_1 = optim.Adam(model.parameters(), lr=inner_lr,
                           weight_decay=train_params['weight_decay'])
    optimizer_2 = optim.Adam(metirc_model.parameters(), lr=inner_lr,
                             weight_decay=train_params['weight_decay'])

    outer_optimizer = optim.Adam(model.parameters(), lr=train_params['outer_lr'],
                                 weight_decay=train_params['weight_decay'])
    metric_optimizer = optim.Adam(metirc_model.parameters(), lr=train_params['metric_lr'],
                                  weight_decay=train_params['weight_decay'])
    # inner_lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    # 定义loss
    dice_crtierion = SoftDiceLoss()
    triplet_loss = TripletSemihardLoss()

    best_val_dice = 0
    best_val_step = 0
    # 包装迭代器
    num_training_tasks = len(loader_train_list)
    num_meta_train = train_params['num_meta_train']  # num_training_tasks-1
    num_meta_test = train_params['num_meta_test']  # as setting num_meta_test = 1
    loader_iter_list = [iter(v) for v in loader_train_list]
    model.train()
    for iter_num in tqdm(range(train_params['start_iter'], train_params['iterations'])):
        # Randomly choosing meta train and meta test domains
        task_list = np.random.permutation(num_training_tasks)
        meta_train_index_list = task_list[:num_meta_train]
        meta_test_index_list = task_list[-num_meta_test:]

        for i in range(num_meta_train):
            task_ind = meta_train_index_list[i]
            if i == 0:
                loader_iter_a = loader_iter_list[task_ind]
                try:
                    batch_a = next(loader_iter_a)
                except StopIteration:
                    loader_iter_a = iter(loader_train_list[task_ind])
                    batch_a = next(loader_iter_a)
            elif i == 1:
                loader_iter_b = loader_iter_list[task_ind]
                try:
                    batch_b = next(loader_iter_b)
                except StopIteration:
                    loader_iter_b = iter(loader_train_list[task_ind])
                    batch_b = next(loader_iter_b)
            else:
                raise RuntimeError('check number of meta-train domains.')

        for i in range(num_meta_test):
            task_ind = meta_test_index_list[i]
            if i == 0:
                loader_iter_t = loader_iter_list[task_ind]
                try:
                    batch_t = next(loader_iter_t)
                except StopIteration:
                    loader_iter_t = iter(loader_train_list[task_ind])
                    batch_t = next(loader_iter_t)
            else:
                raise RuntimeError('check number of meta-test domains.')

        # clone model
        clone_model = maml_model.clone()

        # 训练前向传播
        img_a, gt_a = batch_a['img'].cuda(), batch_a['gt'].cuda()
        img_b, gt_b = batch_b['img'].cuda(), batch_b['gt'].cuda()
        img_t, gt_t = batch_t['img'].cuda(), batch_t['gt'].cuda()

        input_group = torch.cat((img_a[:2], img_b[:1], img_t[:2]), dim=0)
        label_group = torch.cat((img_a[:2], img_b[:1], img_t[:2]), dim=0)

        contour_group, metric_label_group = get_coutour_sample(label_group)

        # Obtaining the conventional task loss on meta-train
        pred_a, _, _ = clone_model(img_a)
        task_lossa = dice_crtierion(pred_a, gt_a, train_params['num_classes'])
        pred_b, _, _ = clone_model(img_b)
        task_lossb = dice_crtierion(pred_b, gt_b, train_params['num_classes'])
        task_loss = (task_lossa + task_lossb) / 2.0
        clone_model.adapt(task_loss)

        ## compute compactness loss
        task_outputt, task_predmaskt, _ = clone_model(img_t)
        task_losst = dice_crtierion(task_outputt, gt_t)
        compactness_loss_t, length, area, boundary_t = get_compactness_cost(task_outputt, gt_t)
        compactness_loss_t = train_params['compactness_loss_weight'] * compactness_loss_t

        # compute smoothness loss
        _, _, embeddings = clone_model(input_group)
        coutour_embeddings = extract_coutour_embedding(contour_group.float(), embeddings)
        metric_embeddings = metirc_model(coutour_embeddings)

        # print(metric_label_group.shape)
        # print(metric_embeddings.shape)
        smoothness_loss_t = triplet_loss(embeddings=metric_embeddings, target=metric_label_group[..., 0],
                                         margin=train_params['margin'])
        smoothness_loss_t = train_params['smoothness_loss_weight'] * smoothness_loss_t

        # 计算loss
        # source_loss = task_loss
        target_loss = task_losst + compactness_loss_t + smoothness_loss_t

        # 反向传播更新参数
        pred_a, _, _ = clone_model(img_a)
        task_lossa = dice_crtierion(pred_a, gt_a, train_params['num_classes'])
        pred_b, _, _ = clone_model(img_b)
        task_lossb = dice_crtierion(pred_b, gt_b, train_params['num_classes'])
        source_loss = (task_lossa + task_lossb) / 2.0

        optimizer_1.zero_grad()
        optimizer_2.zero_grad()

        source_loss.backward()
        smoothness_loss_t.backward(retain_graph=True)
        optimizer_1.step()
        optimizer_2.step()

        outer_optimizer.zero_grad()
        metric_optimizer.zero_grad()
        target_loss.backward(retain_graph=True)
        # smoothness_loss_t.backward()
        nn.utils.clip_grad_norm(metirc_model.parameters(), max_norm=train_params['gradients_clip_value'],
                                norm_type=2)
        metric_optimizer.step()
        outer_optimizer.step()
        # inner_lr_scheduler.step()

        # tensorboard信息
        # writer.add_scalar('metatrain/inner_lr', inner_lr_scheduler.get_lr(), iter_num)
        writer.add_scalar('metatrain/loss/source1_loss', task_lossa, iter_num)
        writer.add_scalar('metatrain/loss/source2_loss', task_lossb, iter_num)
        writer.add_scalar('metatrain/loss/target_loss', task_losst, iter_num)
        writer.add_scalar('metatrain/loss/target_coutour_loss', compactness_loss_t, iter_num)
        writer.add_scalar('target_length', length, iter_num)
        writer.add_scalar('target_area', area, iter_num)

        # 打印训练过程的loss值
        if iter_num % train_params['print_freq'] == 0:
            logging.info('(Iteration %d, inner_lr: %.4f) --> task_loss_a: %.4f; task_loss_b: %.4f; target_loss: %.4f'
                         % (iter_num, inner_lr, task_lossa.item(), task_lossb.item(), task_losst.item()))

        # Validation online
        if iter_num % train_params['val_freq'] == 0:
            # calculate the dice
            loss_bce_val, loss_dice_val, mean_dice_val = _val_on_the_fly(maml_model, loader_val, writer, iter_num)
            logging.info('Validation --> loss_bce: %.4f; loss_dice: %.4f; mean_dice: %.4f' %
                         (loss_bce_val.item(), loss_dice_val.item(), mean_dice_val.item()))

            if mean_dice_val > best_val_dice:
                best_val_dice = mean_dice_val
                best_val_step = iter_num
                torch.save(maml_model.state_dict(), os.path.join(train_params['snapshot_path']+data_params['unseen_site'], 'best_model.pth'))
                logging.info('********** Best model (dice: %.4f) is updated at step %d.' %
                             (mean_dice_val.item(), iter_num))
            else:
                logging.info('********** Best model (dice: %.4f) was at step %d, current dice: %.4f.' %
                             (best_val_dice, best_val_step, mean_dice_val.item()))

        # 保存模型
        if iter_num % train_params['save_model_freq'] == 0:
            save_mode_path = os.path.join(train_params['snapshot_path']+data_params['unseen_site'],
                                          'iter_%d_dice_%.4f.pth' % (iter_num, mean_dice_val.item()))
            torch.save(maml_model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        # # 改变学习率
        # if iter_num % train_params['lr_decay_freq'] == 0:
        #     lr_ = train_params['learning_rate'] * 0.95 ** (iter_num // train_params['lr_decay_freq'])
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr_


if __name__ == '__main__':
    # 配置文件
    settings = Settings()
    common_params, data_params, net_params, train_params, eval_params = settings['COMMON'], settings['DATA'], settings[
        'NETWORK'], settings['TRAINING'], settings['EVAL']

    # 新建任务文件夹
    if common_params['del_exp']:
        shutil.rmtree(common_params['exp_dir']+data_params['unseen_site'], ignore_errors=True)
        os.makedirs(common_params['exp_dir']+data_params['unseen_site'], exist_ok=True)
    logging.basicConfig(filename=os.path.join(common_params['exp_dir']+data_params['unseen_site'], 'logs.txt'),
                        level=logging.DEBUG, format='%(asctime)s %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info('Output path = %s' % common_params['exp_dir']+data_params['unseen_site'])

    # set data list
    source_list = ['HK', 'ISBI', 'ISBI_1.5', 'I2CVB', 'UCL', 'BIDMC']
    source_list.remove(data_params['unseen_site'][:-1])
    train_file_list = [os.path.join(data_params['data_list_dir'], source_domain+'_train_list') for source_domain in source_list]
    valid_file_list = os.path.join(data_params['data_list_dir'], data_params['unseen_site'][:-1]+'_train_list')
    # 配置训练和测试过程的Dataset
    tr_data_list, train_iter_list = [], []
    for i in range(len(train_file_list)):
        tr_data = Meta_Dataset(npz_dir=data_params['npz_data_dir'], datalist_dir=train_file_list[i], mode='train')
        tr_loader = DataLoader(dataset=tr_data, batch_size=train_params['meta_batch_size'],
                               shuffle=True, num_workers=0, drop_last=True, pin_memory=True)
        train_iter_list.append(tr_loader)

    valid_dataset = Meta_Dataset(npz_dir=data_params['npz_data_dir'], datalist_dir=valid_file_list, mode='val')
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True,
                              pin_memory=True)
    # test_dataset = MultiDataset(file_dirs=data_params['data_root_dir'], mode='test')  # 用于测试模型
    logging.info('Using MS-site data to do experiment')

    # 配置 summarywriter
    writer = SummaryWriter(log_dir=common_params['exp_dir']+data_params['unseen_site'])

    # 配置分割网络
    model = forward_Unet(net_params).cuda()
    metric_model = forward_metric_net().cuda()

    train(model, metric_model, train_iter_list, valid_loader, train_params, writer)
