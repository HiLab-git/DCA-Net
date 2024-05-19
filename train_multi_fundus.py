#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/5/28 12:20
# @Author  : Ran. Gu
'''
测试使用domain adaptor来解决multi-site fundus的问题，测试是否比直接joint train效果会更好
'''
import os
import math
from unicodedata import normalize
import torch
import shutil
import random
import timeit
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from settings import Settings
from datetime import datetime
from torchvision.utils import make_grid
###
import Datasets.Mul_fundus.fundus_dataloader as FD
import Datasets.Mul_fundus.custom_transforms as TR
from utils.losses import get_soft_label, DiceLoss, val_dice, val_dice_class
from utils.losses import get_compactness_cost, connectivity_region_analysis
from utils.util import compute_sdf
from utils.mmd import mmd
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchvision import transforms
from utils.Bigaug import fourier_aug

from Models.networks.genda_net import Gen_Domain_Atten_Unet

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
    with torch.no_grad():
        for num, sample in tqdm(enumerate(loader_val), total=len(loader_val),
                                           desc='Valid iteration=%d' % iter_num, ncols=80, leave=False):

                image = sample['image']
                label = sample['label']
                # domain_code = sample['dc']
                img_val = image.cuda()
                gt_val = label.cuda()
                # domain_code = domain_code.cuda()
                with torch.no_grad():
                    pred_val, _ = model(img_val)
                loss_bce_val += bce_criterion(torch.sigmoid(pred_val), gt_val)
                if torch.isnan(loss_bce_val):
                    raise ValueError('loss_bce is nan while validating')
                pred_val = torch.sigmoid(pred_val) > 0.75
                pred_val = connectivity_region_analysis(pred_val.permute(0, 2, 3, 1))
                pred_val = pred_val.permute(0, 3, 1, 2)
                mean_dice_val += val_dice_class(pred_val, gt_val, train_params['num_classes'])
                if num % train_params['print_freq'] == 0:
                    # summarywriter image
                    grid_image = make_grid(img_val.clone().cpu().data, train_params['batch_size'], normalize=True)
                    writer.add_image('val/images', grid_image, iter_num)
                    writer.add_images('val/cup_ground_truths', gt_val[:, 0:1, :, :], iter_num)
                    writer.add_images('val/cup_preds', pred_val[:, 0:1, :, :], iter_num)
                    writer.add_images('val/disc_ground_truths', gt_val[:, 1:2, :, :], iter_num)
                    writer.add_images('val/disc_preds', pred_val[:, 1:2, :, :], iter_num)
                    
        loss_bce_val /= len(loader_val)
        mean_dice_val /= len(loader_val)
        loss_dice_val = 1 - torch.mean(mean_dice_val)
        # summarywriter
        writer.add_scalar('val/loss_bce', loss_bce_val, iter_num)
        writer.add_scalars('val/mean_dice', {'cup': mean_dice_val[0], 'disc': mean_dice_val[1]}, iter_num)
        
        return loss_bce_val, loss_dice_val, mean_dice_val


def train(model: torch.nn.Module, loader_train: list, loader_val: DataLoader, train_params: dict, writer):
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=train_params['learning_rate'],
                           weight_decay=train_params['weight_decay'])

    # 定义loss
    mse_criterion = nn.MSELoss()
    bce_criterion = nn.BCELoss()
    dice_crtierion = DiceLoss()
    # cos_crtierion = nn.CosineSimilarity(dim=0)

    lr_ = train_params['learning_rate']

    # restrain the model if start_iter > 0
    if train_params['resume'] and train_params['start_iter'] > 0:
        log_dir = train_params['snapshot_path'] + 'domain'+str(data_params['datasettest']) + 'logs.txt'
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
        iter_list = [x for x in os.listdir(train_params['snapshot_path'] + 'domain'+str(data_params['datasettest']))
                     if x.startswith('n' + str(net_params['num_adaptor']))]
        for iter_name in iter_list:
            if e_modelname in iter_name.split('_dice_')[0]:
                modelname = os.path.join(train_params['snapshot_path'] + 'domain'+str(data_params['datasettest']), iter_name)
                print("=> Loading checkpoint '{}'".format(modelname))
                checkpoint = torch.load(modelname)
                pretrained_dict = checkpoint['model_dict']
                model_dict = model.state_dict()
                # 1. filter out unnecessary keys
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                # 2. overwrite entries in the existing state dict
                model_dict.update(pretrained_dict)
                # 3. load the new state dict
                model.load_state_dict(model_dict)
                # optimizer.load_state_dict(checkpoint['opt_dict'])
                print("=> Loaded the saved model checkpoint.")
            else:
                print(
                    "=> No checkpoint found at '{}'".format(train_params['snapshot_path'] + data_params['unseen_site']))
    
    best_val_dice = 0
    best_val_cup_dice = 0
    best_val_disc_dice = 0
    best_val_step = 0
    
    # start_time = timeit.default_timer()
    for epoch in tqdm(range(train_params['start_epoch'], train_params['num_epoch']),
                      initial=train_params['start_epoch'], total=train_params['num_epoch'], ncols=80, leave=False):

        for batch_idx, sample in tqdm(enumerate(loader_train), total=len(loader_train),
                                      desc='Train epoch=%d' % epoch, ncols=80, leave=False):
            
            iter_num = batch_idx + epoch * len(loader_train)
            model.train()
            optimizer.zero_grad()

            image = None
            img_name = None
            label = None
            img_gan = None
            img_fft = None
            domain_code = None
            ganTrue = None
            for domain in sample:
                if image is None:
                    image = domain['image']
                    img_name = domain['img_name']
                    img_gan = domain['img_gan']
                    img_fft = domain['img_gan']
                    label = domain['label']
                    domain_code = domain['dc']
                    ganTrue = domain['ganTrue']
                else:
                    image = torch.cat([image, domain['image']], 0)
                    img_name = np.concatenate((img_name, domain['img_name']), axis = 0)
                    ganTrue = np.concatenate((ganTrue, domain['ganTrue']), axis = 0)
                    img_gan = torch.cat([img_gan, domain['img_gan']], 0)
                    img_fft = torch.cat([img_fft, domain['img_gan']], 0)
                    label = torch.cat([label, domain['label']], 0)
                    domain_code = torch.cat([domain_code, domain['dc']], 0)

            # image_aug = torch.zeros(train_params['aug_num'], image.shape[0], image.shape[1], image.shape[2], image.shape[3])
            image = image.cuda()
            target_map = label.cuda()
            img_gan = img_gan.cuda()
            random_arr = []
            for i in range(len(ganTrue)):
                if ganTrue[i] != '0':
                    random_arr.append(i)
            for bs in range(img_fft.shape[0]):
                if ganTrue[bs] != '0':
                    random_p = random.choice(random_arr)
                    img_fft[bs] = fourier_aug(img_fft[bs],img_fft[random_p])
            img_fft = img_fft.cuda()
            seg_pred, domain_feature = model(image)
            seg_pred_gan, _ = model(img_gan)
            seg_pred_fft, _ = model(img_fft)
            consistency_loss = mse_criterion(torch.sigmoid(seg_pred_gan), torch.sigmoid(seg_pred_fft))
            del seg_pred_fft
            del seg_pred_gan
            # 计算loss
            loss_domain_sdf_all = 0
            for i in range(len(domain_feature)):
                loss_domain_sdf = 0
                loss_domain_sdf += mmd(domain_feature[i][0].squeeze(), domain_feature[i][1].squeeze())
                loss_domain_sdf += mmd(domain_feature[i][1].squeeze(), domain_feature[i][2].squeeze())
                loss_domain_sdf += mmd(domain_feature[i][0].squeeze(), domain_feature[i][2].squeeze())
                loss_domain_sdf_all += 10**-i * (1 - torch.sigmoid(loss_domain_sdf / len(domain_feature[i])))

            compact_loss, _, _, _ = get_compactness_cost(torch.sigmoid(seg_pred), target_map)
            loss_bce = bce_criterion(torch.sigmoid(seg_pred), target_map)
            loss_dice = dice_crtierion(torch.sigmoid(seg_pred), target_map, train_params['num_classes'])            
            if torch.isnan(loss):
                raise ValueError('loss is nan while training')

            loss = loss_dice + 0.1*loss_domain_sdf_all + compact_loss + consistency_loss
            # 反向传播更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # tensorboard信息
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss_bce', loss_bce, iter_num)
            writer.add_scalar('loss/loss_dice', loss_dice, iter_num)
            writer.add_scalar('loss/cos_simila', loss_domain_sdf_all, iter_num)
            writer.add_scalar('loss/compact_loss', compact_loss, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)

            grid_image = make_grid(image.clone().cpu().data, train_params['batch_size'], normalize=True)
            writer.add_image('train/images', grid_image, iter_num)
            grid_image = make_grid(target_map[:, 0:1, ...].clone().cpu().data, train_params['batch_size'], normalize=True)
            writer.add_image('train/cup_ground_truths', grid_image, iter_num)
            grid_image = make_grid(torch.sigmoid(seg_pred)[:, 0:1, ...].clone().cpu().data, train_params['batch_size'], normalize=True)
            writer.add_image('train/cup_preds', grid_image, iter_num)
            grid_image = make_grid(target_map[:, 1:2, ...].clone().cpu().data, train_params['batch_size'], normalize=True)
            writer.add_image('train/disc_ground_truths', grid_image, iter_num)
            grid_image = make_grid(torch.sigmoid(seg_pred)[:, 1:2, ...].clone().cpu().data, train_params['batch_size'], normalize=True)
            writer.add_image('train/disc_preds', grid_image, iter_num)

            # 打印训练过程的loss值
            if iter_num % train_params['print_freq'] == 0:
                logging.info('\n(Iteration %d, lr: %.4f) --> loss_bce: %.4f; loss_dice: %.4f; loss: %.4f; loss_hausdorff: %.4f; loss_sdf: %.4f'
                            % (iter_num, lr_, loss_bce.item(), loss_dice.item(), loss.item(), loss_domain_sdf_all, compact_loss.item()))

            # Validation online
            if iter_num % train_params['val_freq'] == 0 and iter_num > 0:
                # calculate the dice
                loss_bce_val, loss_dice_val, mean_dice_val = _val_on_the_fly(model, loader_val, writer, iter_num)
                logging.info('\nValidation --> loss_bce: %.4f; loss_dice: %.4f; mean_dice_cup: %.4f; mean_dice_disc: %.4f' %
                            (loss_bce_val.item(), loss_dice_val.item(), mean_dice_val[0].item(), mean_dice_val[1].item()))

                if torch.mean(mean_dice_val) > best_val_dice:
                    best_val_dice = torch.mean(mean_dice_val)
                    best_val_cup_dice = mean_dice_val[0]
                    best_val_disc_dice = mean_dice_val[1]
                    best_val_step = iter_num
                    torch.save(model.state_dict(), os.path.join(train_params['snapshot_path']+'domain'+str(data_params['datasettest'][0]), 'n' +
                                                                str(net_params['num_adaptor'])+'best_model'+'.pth'))
                    logging.info('********** Best model (dice: %.4f; cup_dice: %.4f; disc_dice: %.4f) is updated at step %d.' %
                                (torch.mean(mean_dice_val).item(), mean_dice_val[0].item(), mean_dice_val[1].item(), iter_num))
                else:
                    logging.info('********** Best model (dice: %.4f; cup_dice: %.4f; disc_dice: %.4f) was at step %d, current dice: %.4f.' %
                                (best_val_dice, best_val_cup_dice, best_val_disc_dice, best_val_step, torch.mean(mean_dice_val).item()))

            # 保存模型
            if iter_num % train_params['save_model_freq'] == 0 and iter_num > 0:
                save_mode_path = os.path.join(train_params['snapshot_path']+'domain'+str(data_params['datasettest'][0]), 'n' +
                                            str(net_params['num_adaptor'])+'_iter_%d_dice_%.4f.pth'
                                            % (iter_num, torch.mean(mean_dice_val).item()))
                torch.save({'model_dict': model.state_dict(), 
                            'optim_dict': optimizer.state_dict()},
                           save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            # 改变学习率
            if iter_num > train_params['lr_frozen'] and iter_num % train_params['lr_decay_freq'] == 0:
                lr_ = train_params['learning_rate'] * 0.95 ** ((iter_num - train_params['lr_frozen'])  // train_params['lr_decay_freq'])
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

if __name__ == '__main__':
    # 配置文件
    settings = Settings()
    common_params, data_params, net_params, train_params, eval_params = settings['COMMON'], settings['DATA'], settings[
        'NETWORK'], settings['TRAINING'], settings['EVAL']

    # 新建任务文件夹
    if common_params['del_exp']:
        shutil.rmtree(common_params['exp_dir']+'domain'+str(data_params['datasettest'][0]), ignore_errors=True)
        os.makedirs(common_params['exp_dir']+'domain'+str(data_params['datasettest'][0]), exist_ok=True)
    logging.basicConfig(filename=os.path.join(common_params['exp_dir']+'domain'+str(data_params['datasettest'][0]), 'logs.txt'),
                        level=logging.DEBUG, format='%(asctime)s %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info('Output path = %s' % common_params['exp_dir']+'domain'+str(data_params['datasettest'][0]))

    # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    cuda = torch.cuda.is_available()
    torch.cuda.manual_seed(1337)

    # set data list
    composed_transforms_tr = transforms.Compose([
        TR.RandomScaleCrop(512),
        TR.Normalize_tf(512),
        TR.ToTensor()
    ])

    composed_transforms_ts = transforms.Compose([
        TR.RandomCrop(256),
        TR.Normalize_tf(),
        TR.ToTensor()
    ])

    # 配置训练和测试过程的Dataset
    domain = FD.FundusSegmentation(base_dir=data_params['data_dir'], phase='train', splitid=data_params['datasettrain'],
                                   testid=data_params['datasettest'], transform=composed_transforms_tr)
    train_loader = DataLoader(domain, batch_size=train_params['batch_size'], shuffle=True, num_workers=4, pin_memory=True)

    domain_val = FD.FundusSegmentation(base_dir=data_params['data_dir'], phase='test', splitid=data_params['datasettest'],
                                       testid=data_params['datasettest'], transform=composed_transforms_ts)
    valid_loader = DataLoader(domain_val, batch_size=train_params['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

    # 配置 summarywriter
    writer = SummaryWriter(log_dir=common_params['exp_dir'] + 'domain' + str(data_params['datasettest'][0]))

    # 配置分割网络
    model = Gen_Domain_Atten_Unet(net_params).cuda()
    print('parameter numer:', sum([p.numel() for p in model.parameters()]))

    if train_params['resume']:
        checkpoint = torch.load(train_params['resume'])
        pretrained_dict = checkpoint['model_state_dict']
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)

        print('Loaded the pretrained model at: ', train_params['resume'])
        # model.freeze_para()

    train(model, train_loader, valid_loader, train_params, writer)
