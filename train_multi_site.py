#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/10/23 20:11
# @Author  : Ran. Gu
'''
测试使用domain adaptor来解决multi-site的问题，测试是否比直接joint train效果会更好
'''
import os
import math
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
from Datasets.Mul_prostate.MultiDataset import MultiDataset
from Datasets.Mul_prostate.Meta_Dataset import Meta_Dataset
from Datasets.Mul_prostate.Meta_Multi_Dataset import Meta_Multi_Dataset
from Datasets.Mul_prostate.Meta_Multi_Dataset import RandomCrop, RandomNoise, RandomRotFlip, ToTensor
from utils.losses import get_soft_label, SoftDiceLoss, val_dice, val_dice_class, connectivity_region_analysis
from utils.losses import get_compactness_cost
from utils.util import compute_sdf
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchvision import transforms

# from Models.networks.genda_net import Gen_Domain_Atten_Unet
# from Models.networks.da_net import Domain_Atten_Unet
from Models.networks.genda_net_ds import Gen_Domain_Atten_Unet

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

### 实验重复性
# torch.manual_seed(123)
# torch.cuda.manual_seed(123)
# torch.cuda.manual_seed_all(123)
# np.random.seed(123)  # Numpy module.
# random.seed(123)  # Python random module.
# torch.manual_seed(123)
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
        img_val, gt_val = batch_val['image'].cuda(), batch_val['label'].cuda()
        with torch.no_grad():
            pred_val, _, _ = model(img_val, training=False)
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


def train(model: torch.nn.Module, loader_train: list, loader_val: DataLoader, train_params: dict, writer):
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=train_params['learning_rate'],
                           weight_decay=train_params['weight_decay'])

    # 定义loss
    mse_criterion = nn.MSELoss()
    bce_criterion = nn.BCELoss()
    dice_crtierion = SoftDiceLoss()
    # cos_crtierion = nn.CosineSimilarity(dim=0)

    # 包装迭代器
    loader_iter = [iter(v) for v in loader_train]

    lr_ = train_params['learning_rate']
    best_val_dice = 0
    best_val_step = 0
    # restrain the model if start_iter > 0
    if train_params['start_iter'] > 0:
        log_dir = train_params['snapshot_path'] + data_params['unseen_site'] + 'logs.txt'
        with open(log_dir, 'r') as f:
            lines = f.readlines()
            lines.reverse()
            for line in lines:
                if 'Best model' in line:
                    best_val_dice = float(line[53:59])
                    best_val_step = int(line.split('step ')[-1].split(',')[0])
                    break

        # Load the trained best model
        e_modelname = 'n' + str(net_params['num_adaptor']) + '_iter_' + str(train_params['start_iter'])
        iter_list = [x for x in os.listdir(train_params['snapshot_path'] + data_params['unseen_site'])
                     if x.startswith('n' + str(net_params['num_adaptor']))]
        for iter_name in iter_list:
            if e_modelname in iter_name.split('_dice_')[0]:
                modelname = os.path.join(train_params['snapshot_path'] + data_params['unseen_site'], iter_name)
                print("=> Loading checkpoint '{}'".format(modelname))
                checkpoint = torch.load(modelname)
                model.load_state_dict(checkpoint)
                # optimizer.load_state_dict(checkpoint['opt_dict'])
                print("=> Loaded the saved model checkpoint.")
            else:
                print(
                    "=> No checkpoint found at '{}'".format(train_params['snapshot_path'] + data_params['unseen_site']))

    for iter_num in tqdm(range(train_params['start_iter'], train_params['iterations']),
                         initial=train_params['start_iter'], total=train_params['iterations']):
        img, gt = [], []
        random.shuffle(loader_iter)
        for i in range(len(train_file_list)):
            try:
                batch = next(loader_iter[i])
            except StopIteration:
                loader_iter[i] = iter(loader_train[i])
                batch = next(loader_iter[i])
            # print(batch['slice_name'])
            img.append(batch['image'])
            gt.append(batch['label'])

        img = torch.cat(img, dim=0).cuda()
        gt = torch.cat(gt, dim=0).cuda()
        # 训练前向传播
        model.train()
        # img, gt = batch['img'].cuda(), batch['gt'].cuda()
        seg_pred, seg_tanh, domain_feature = model(img, training=True)

        # 计算loss
        # with torch.no_grad():
        #     gt_dis = compute_sdf(torch.argmax(gt, dim=1, keepdim=True).cpu().numpy(), domain1_c[:, 1, ...].shape)
        #     gt_dis = torch.from_numpy(gt_dis).float().cuda()
        # loss_sdf = mse_criterion(domain1_c[:, 1, ...], gt_dis)
        # loss_sdf2 = mse_criterion(dof2_dm[:, 1, ...], gt_dis)
        loss_domain_sdf_all = 0
        for i in range(len(domain_feature)):
            loss_domain_sdf = 0
            loss_domain_sdf += mse_criterion(domain_feature[i][0], domain_feature[i][1])
            loss_domain_sdf += mse_criterion(domain_feature[i][1], domain_feature[i][2])
            loss_domain_sdf += mse_criterion(domain_feature[i][0], domain_feature[i][2])
            loss_domain_sdf_all += 10**-i * (1 - loss_domain_sdf / len(domain_feature[i]))

        compact_loss, _, _, _ = get_compactness_cost(seg_pred, gt)
        # loss_cosine = torch.mean(abs(cos_crtierion(dof1_c, dof2_c)))

        loss_bce = bce_criterion(seg_pred, gt)
        loss_dice = dice_crtierion(seg_pred, gt, train_params['num_classes'])

        # domain specific similarity loss
        # domain_sta_1 = []
        # domain_sta_2 = []
        # domain_silimar_loss = 0
        # choose_domain = random.sample(range(0, net_params['num_adaptor']), net_params['num_adaptor'])
        # for do in range(0, net_params['num_adaptor'], 2):
        #     domain_sta_1 = domain_sta[:, choose_domain[do]:choose_domain[do]+1, ...]
        #     domain_sta_2 = domain_sta[:, choose_domain[do+1]:choose_domain[do+1]+1, ...]
        #     domain_silimar_loss += mse_criterion(domain_sta_1, domain_sta_2)
        # loss_similarity = 1 - domain_silimar_loss / (net_params['num_adaptor']//2)
        # domain_sta_1 = torch.cat(domain_sta_1, dim=1)
        # domain_sta_2 = torch.cat(domain_sta_2, dim=1)
        # loss_similarity = mse_criterion(domain_sta_1, domain_sta_2)

        loss = 0.5*(loss_bce+loss_dice) + 0.1*loss_domain_sdf_all + compact_loss
        # loss_sdf = 0.5*(loss_sdf1+loss_sdf2) + 0.2*loss_cosine
        # 反向传播更新参数
        optimizer.zero_grad()
        loss.backward()
        # compact_loss.backward()
        # loss_sdf.backward()
        optimizer.step()

        # tensorboard信息
        writer.add_scalar('lr', lr_, iter_num)
        writer.add_scalar('loss/loss_bce', loss_bce, iter_num)
        writer.add_scalar('loss/loss_dice', loss_dice, iter_num)
        writer.add_scalar('loss/cos_simila', loss_domain_sdf, iter_num)
        # writer.add_scalar('loss/loss_sdf', loss_sdf, iter_num)
        writer.add_scalar('loss/compact_loss', compact_loss, iter_num)
        # writer.add_scalar('loss/loss_hausdorff', loss_domain_sdf, iter_num)
        writer.add_scalar('loss/loss', loss, iter_num)

        # 打印训练过程的loss值
        if iter_num % train_params['print_freq'] == 0:
            logging.info('(Iteration %d, lr: %.4f) --> loss_bce: %.4f; loss_dice: %.4f; loss: %.4f; loss_hausdorff: %.4f; loss_sdf: %.4f'
                         % (iter_num, lr_, loss_bce.item(), loss_dice.item(), loss.item(), loss_domain_sdf, compact_loss.item()))

        # Validation online
        if iter_num % train_params['val_freq'] == 0 and iter_num > 0:
            # calculate the dice
            loss_bce_val, loss_dice_val, mean_dice_val = _val_on_the_fly(model, loader_val, writer, iter_num)
            logging.info('Validation --> loss_bce: %.4f; loss_dice: %.4f; mean_dice: %.4f' %
                         (loss_bce_val.item(), loss_dice_val.item(), mean_dice_val.item()))

            if mean_dice_val > best_val_dice:
                best_val_dice = mean_dice_val
                best_val_step = iter_num
                torch.save(model.state_dict(), os.path.join(train_params['snapshot_path']+data_params['unseen_site'],
                                                            'best_model_'+str(net_params['num_adaptor'])+'.pth'))
                logging.info('********** Best model (dice: %.4f) is updated at step %d.' %
                             (mean_dice_val.item(), iter_num))
            else:
                logging.info('********** Best model (dice: %.4f) was at step %d, current dice: %.4f.' %
                             (best_val_dice, best_val_step, mean_dice_val.item()))

        # 保存模型
        if iter_num % train_params['save_model_freq'] == 0 and iter_num > 0:
            save_mode_path = os.path.join(train_params['snapshot_path']+data_params['unseen_site'], 'n' +
                                          str(net_params['num_adaptor'])+'_iter_%d_dice_%.4f.pth'
                                          % (iter_num, mean_dice_val.item()))
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        # 改变学习率
        # if iter_num % train_params['lr_decay_freq'] == 0:
        #     lr_ = train_params['learning_rate'] * 0.5 ** (iter_num // train_params['lr_decay_freq'])
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
    train_file_list = [os.path.join(data_params['data_list_dir'], source_domain + '_train_list') for source_domain in
                       source_list]
    valid_file_list = [os.path.join(data_params['data_list_dir'], data_params['unseen_site'][:-1] + '_train_list')]
    train_dataset = [Meta_Multi_Dataset(npz_dir=data_params['npz_data_dir'], datalist_list=site_list, mode='train',
                                        transform=transforms.Compose([
                                            # RandomRotFlip(),
                                            # RandomNoise(),
                                            ToTensor(),
                                        ])) for site_list in train_file_list]
    valid_dataset = Meta_Multi_Dataset(npz_dir=data_params['npz_data_dir'], datalist_list=valid_file_list[0], mode='val',
                                       transform=transforms.Compose([
                                           ToTensor(),
                                       ]))
    # 配置训练和测试过程的Dataset
    # train_dataset = MultiDataset(npz_dir=data_params['npz_data_dir'], datalist_dir=data_params['data_list_dir'],
    #                              ignore_site=data_params['unseen_site'], mode='train')
    # valid_dataset = MultiDataset(npz_dir=data_params['npz_data_dir'], datalist_dir=data_params['data_list_dir'],
    #                              ignore_site=data_params['unseen_site'], mode='val')
    # test_dataset = MultiDataset(file_dirs=data_params['data_root_dir'], mode='test')  # 用于测试模型
    logging.info('Using MS-site data to do experiment')
    train_loader = [DataLoader(dataset=t_set, batch_size=train_params['batch_size'],
                               shuffle=True, num_workers=0, drop_last=True, pin_memory=True) for t_set in train_dataset]
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=train_params['batch_size'],
                              shuffle=True, num_workers=0, drop_last=True, pin_memory=True)

    # 配置 summarywriter
    writer = SummaryWriter(log_dir=common_params['exp_dir']+data_params['unseen_site'])

    # 配置分割网络
    model = Gen_Domain_Atten_Unet(net_params).cuda()
    # model = nn.DataParallel(model)
    # model = Domain_Atten_Unet(net_params).cuda()
    # model = nn.DataParallel(model)

    train(model, train_loader, valid_loader, train_params, writer)
