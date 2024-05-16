#!/usr/bin/env python

import argparse
import os
import os.path as osp
import torch.nn.functional as F

import random
import torch
import logging
from settings import Settings
from torch.autograd import Variable
import tqdm
from Datasets.Mul_fundus import fundus_dataloader_raw as DL
from torch.utils.data import DataLoader
from Datasets.Mul_fundus import custom_transforms_raw as tr
from torchvision import transforms
from Datasets import utils
# from scipy.misc import imsave
from utils.Utils_for_fundus import joint_val_image, postprocessing, save_per_img
from utils.metrics import *
from utils.losses import val_dice_class
from datetime import datetime
import pytz
import cv2
import numpy as np
from medpy.metric import binary
from Models.networks.genda_net_ds import Gen_Domain_Atten_Unet
from Models.networks.deeplabv3 import DeepLab
from utils.Bigaug import fourier_interpolation, fourier_interpolation_amp, augment_gamma, augment_gaussian_noise, fourier_aug, augment_rot90, augment_mirroring,augment_transpose_axes,augment_rot90_reverse

def construct_color_img(prob_per_slice):
    shape = prob_per_slice.shape
    img = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    img[:, :, 0] = prob_per_slice * 255
    img[:, :, 1] = prob_per_slice * 255
    img[:, :, 2] = prob_per_slice * 255

    im_color = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    return im_color


def normalize_ent(ent):
    '''
    Normalizate ent to 0 - 1
    :param ent:
    :return:
    '''
    max = np.amax(ent)
    # print(max)

    min = np.amin(ent)
    # print(min)
    return (ent - min) / 0.4


def draw_ent(prediction, save_root, name):
    '''
    Draw the entropy information for each img and save them to the save path
    :param prediction: [2, h, w] numpy
    :param save_path: string including img name
    :return: None
    '''
    if not os.path.exists(os.path.join(save_root, 'disc')):
        os.makedirs(os.path.join(save_root, 'disc'))
    if not os.path.exists(os.path.join(save_root, 'cup')):
        os.makedirs(os.path.join(save_root, 'cup'))
    # save_path = os.path.join(save_root, img_name[0])
    smooth = 1e-8
    cup = prediction[0]
    disc = prediction[1]
    cup_ent = - cup * np.log(cup + smooth)
    disc_ent = - disc * np.log(disc + smooth)
    cup_ent = normalize_ent(cup_ent)
    disc_ent = normalize_ent(disc_ent)
    disc = construct_color_img(disc_ent)
    cv2.imwrite(os.path.join(save_root, 'disc', name.split('.')[0]) + '.png', disc)
    cup = construct_color_img(cup_ent)
    cv2.imwrite(os.path.join(save_root, 'cup', name.split('.')[0]) + '.png', cup)


def draw_mask(prediction, save_root, name):
    '''
    Draw the mask probability for each img and save them to the save path
   :param prediction: [2, h, w] numpy
   :param save_path: string including img name
   :return: None
   '''
    if not os.path.exists(os.path.join(save_root, 'disc')):
        os.makedirs(os.path.join(save_root, 'disc'))
    if not os.path.exists(os.path.join(save_root, 'cup')):
        os.makedirs(os.path.join(save_root, 'cup'))
    cup = prediction[0]
    disc = prediction[1]

    disc = construct_color_img(disc)
    cv2.imwrite(os.path.join(save_root, 'disc', name.split('.')[0]) + '.png', disc)
    cup = construct_color_img(cup)
    cv2.imwrite(os.path.join(save_root, 'cup', name.split('.')[0]) + '.png', cup)



def draw_boundary(prediction, save_root, name):
    '''
    Draw the mask probability for each img and save them to the save path
   :param prediction: [2, h, w] numpy
   :param save_path: string including img name
   :return: None
   '''
    if not os.path.exists(os.path.join(save_root, 'boundary')):
        os.makedirs(os.path.join(save_root, 'boundary'))
    boundary = prediction[0]

    boundary = construct_color_img(boundary)
    cv2.imwrite(os.path.join(save_root, 'boundary', name.split('.')[0]) + '.png', boundary)


def main():
    # 配置文件
    settings = Settings()
    common_params, data_params, net_params, train_params, eval_params = settings['COMMON'], settings['DATA'], settings[
        'NETWORK'], settings['TRAINING'], settings['EVAL']

    logging.basicConfig(filename=os.path.join(common_params['exp_dir']+'domain'+str(data_params['datasettest'][0]), 'results.txt'),
                        level=logging.DEBUG, format='%(asctime)s %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info('Output path = %s' % common_params['exp_dir']+'domain'+str(data_params['datasettest'][0]))

    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # model_file = args.model_file
    output_path = os.path.join(common_params['exp_dir']+'domain'+str(data_params['datasettest'][0]), 'test')
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    # 1. dataset
    composed_transforms_tr = transforms.Compose([
        # tr.RandomScaleCrop(256),
        tr.Normalize_tf(),
        tr.ToTensor()
    ])
    composed_transforms_test = transforms.Compose([
        tr.Normalize_tf(),                                                                                                                                                                                                                              
        tr.ToTensor()
    ])

    batch_size = 8
    db_train = DL.FundusSegmentation(base_dir=data_params['data_dir'], phase='train', splitid=data_params['datasettrain'],
                                   testid=data_params['datasettest'], transform=composed_transforms_tr)
    train_loader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    db_test = DL.FundusSegmentation(base_dir=data_params['data_dir'], phase='test', splitid=data_params['datasettest'],
                                    testid=data_params['datasettest'], transform=composed_transforms_test, state='prediction')
    test_loader = DataLoader(db_test, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 2. model
    model = Gen_Domain_Atten_Unet(net_params).cuda()

    if torch.cuda.is_available():
        model = model.cuda()
    print('==> Loading %s model file: %s' %
          (model.__class__.__name__, eval_params['model_file']))
    # model_data = torch.load(model_file)

    checkpoint = torch.load(eval_params['model_file'])
    # pretrained_dict = checkpoint['model_state_dict']
    pretrained_dict = checkpoint
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)

    if eval_params['movingbn']:
        model.train()
    else:
        model.eval()

    val_cup_dice = 0.0
    val_disc_dice = 0.0
    total_hd_OC = 0.0
    total_hd_OD = 0.0
    total_asd_OC = 0.0
    total_asd_OD = 0.0
    total_rec_OC = 0.0
    total_rec_OD = 0.0
    total_spec_OC = 0.0
    total_spec_OD = 0.0
    timestamp_start = datetime.now(pytz.timezone('Asia/Hong_Kong'))
    total_num = 0
    OC = []
    OC_hd = []
    OC_asd = []
    OD = []
    OD_hd = []
    OD_asd = []
    OC_rec = []
    OD_rec = []
    OC_spec = []
    OD_spec = []

    for batch_idx, (sample) in tqdm.tqdm(enumerate(test_loader),total=len(test_loader),ncols=80, leave=False):

        data = sample['image']
        img_aug_raw = sample['image']
        target = sample['label']
        img_name = sample['img_name']

    
        ensemble_pre = 0
        for i in range(train_params['aug_num']):
            torch.cuda.empty_cache()
            sample_p = db_train[0]
            domain_p = i%3
            data_p = sample_p[domain_p]['image'].cpu().numpy()
            img_aug = img_aug_raw.cpu().numpy()

            for bs in range(img_aug.shape[0]):
                img_aug[bs] = fourier_interpolation_amp(img_aug[bs], data_p)

            img_aug = torch.tensor(img_aug)
        
            if torch.cuda.is_available():
                img_aug = img_aug.cuda()
            img_aug = Variable(img_aug)
            pre_aug, _ = model(img_aug)
            pre_aug = pre_aug.cpu().detach().numpy()
            pre_aug = torch.tensor(pre_aug.copy()).cuda()
            ensemble_pre = pre_aug

        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        prediction, _ = model(data, extract_feature=False)
        prediction = ensemble_pre

        prediction = torch.nn.functional.interpolate(prediction, size=(target.size()[2], target.size()[3]), mode="bilinear")
        data = torch.nn.functional.interpolate(data, size=(target.size()[2], target.size()[3]), mode="bilinear")

        target_numpy = target.data.cpu()
        imgs = data.data.cpu()
        hd_OC = 100
        asd_OC = 100
        hd_OD = 100
        asd_OD = 100
        for i in range(prediction.shape[0]):
            prediction_post = postprocessing(prediction[i], dataset=eval_params['dataset'])
            cup_dice, disc_dice = dice_coeff_2label(prediction_post, target[i])
            cup_rec, disc_rec, cup_spec, disc_spec = recall_specificity(prediction_post, target[i])
            
            # class_dice = val_dice_class(torch.from_numpy(np.expand_dims(prediction_post, axis=0)).cuda(), target[i:i+1], num_class=train_params['num_classes'])
            # cup_dice, disc_dice = class_dice[0], class_dice[1]
            OC.append(cup_dice)
            OD.append(disc_dice)
            OC_rec.append(cup_rec)
            OD_rec.append(disc_rec)
            OC_spec.append(cup_spec)
            OD_spec.append(disc_spec)
            if np.sum(prediction_post[0, ...]) < 1e-4:
                hd_OC = 100
                asd_OC = 100
            else:
                hd_OC = binary.hd95(np.asarray(prediction_post[0, ...], dtype=np.bool),
                                    np.asarray(target_numpy[i, 0, ...], dtype=np.bool))
                asd_OC = binary.asd(np.asarray(prediction_post[0, ...], dtype=np.bool),
                                    np.asarray(target_numpy[i, 0, ...], dtype=np.bool))
            if np.sum(prediction_post[1, ...]) < 1e-4:
                hd_OD = 100
                asd_OD = 100
            else:
                hd_OD = binary.hd95(np.asarray(prediction_post[1, ...], dtype=np.bool),
                                    np.asarray(target_numpy[i, 1, ...], dtype=np.bool))

                asd_OD = binary.asd(np.asarray(prediction_post[1, ...], dtype=np.bool),
                                    np.asarray(target_numpy[i, 1, ...], dtype=np.bool))
            OC_hd.append(hd_OC)
            OD_hd.append(hd_OD)
            OC_asd.append(asd_OC)
            OD_asd.append(asd_OD)

            val_cup_dice += cup_dice
            val_disc_dice += disc_dice
            total_hd_OC += hd_OC
            total_hd_OD += hd_OD
            total_asd_OC += asd_OC
            total_asd_OD += asd_OD
            total_rec_OC += cup_rec
            total_rec_OD += disc_rec
            total_spec_OC += cup_spec
            total_spec_OD += disc_spec
            total_num += 1
            
            # save overlay images
            for img, lt, lp in zip([imgs[i]], [target_numpy[i]], [prediction_post]):
                img, lt = utils.untransform(img, lt)
                ###########################################
                cup_pred = lp[0]
                disc_pred = lp[1]
                
                cup_pred_outpath = os.path.join(output_path, 'pre_cup', img_name[i].split('.')[0]+'.png')
                disc_pred_outpath = os.path.join(output_path, 'pre_disc', img_name[i].split('.')[0]+'.png')
                if not os.path.exists(os.path.dirname(cup_pred_outpath)):
                    os.makedirs(os.path.dirname(cup_pred_outpath))
                if not os.path.exists(os.path.dirname(disc_pred_outpath)):
                    os.makedirs(os.path.dirname(disc_pred_outpath))

                cup_img = cup_pred.squeeze().squeeze()
                disc_img = disc_pred.squeeze().squeeze()
                cup_img[cup_img==1]=255
                disc_img[disc_img==1]=255

                cv2.imwrite(cup_pred_outpath, cup_img)
                cv2.imwrite(disc_pred_outpath, disc_img)
                ###########################################
                save_per_img(img.numpy().transpose(1, 2, 0),
                             output_path,
                             img_name[i],
                             lp, lt, mask_path=None, ext="bmp")

    print('OC:', OC)
    print('OD:', OD)
    import csv
    with open(output_path+'/Dice_results.csv', 'w+') as result_file:
        wr = csv.writer(result_file, dialect='excel')
        for index in range(len(OC)):
            wr.writerow([OC[index], OD[index]])

    val_cup_dice /= total_num
    val_cup_dice_std = np.std(OC)
    val_disc_dice /= total_num
    val_disc_dice_std = np.std(OD)
    total_dice = np.mean(OC+OD)
    total_dice_std = np.std(OC+OD)

    total_hd_OC /= total_num
    total_hd_OC_std = np.std(OC_hd)
    total_hd_OD /= total_num
    total_hd_OD_std = np.std(OD_hd)
    total_hd = np.mean(OC_hd+OD_hd)
    total_hd_std = np.std(OC_hd+OD_hd)

    total_asd_OC /= total_num
    total_asd_OC_std = np.std(OC_asd) 
    total_asd_OD /= total_num
    total_asd_OD_std = np.std(OD_asd)
    total_asd = np.mean(OC_asd+OD_asd)
    total_asd_std = np.std(OC_asd+OD_asd)

    total_rec_OC /= total_num
    total_rec_OC_std = np.std(OC_rec)
    total_spec_OC /= total_num
    total_spec_OC_std = np.std(OC_spec)
    total_rec_OD /= total_num
    total_rec_OD_std = np.std(OD_rec)
    total_spec_OD /= total_num
    total_spec_OD_std = np.std(OD_spec)

    print('''\n==>val_cup_dice : {0}-{1}'''.format(val_cup_dice, val_cup_dice_std))
    print('''\n==>val_disc_dice : {0}-{1}'''.format(val_disc_dice, val_disc_dice_std))
    print('''\n==>val_average_dice : {0}-{1}'''.format(total_dice, total_dice_std))
    print('''\n==>ave_hd_OC : {0}-{1}'''.format(total_hd_OC, total_hd_OC_std))
    print('''\n==>ave_hd_OD : {0}-{1}'''.format(total_hd_OD, total_hd_OD_std))
    print('''\n==>ave_average_hd : {0}-{1}'''.format(total_hd, total_hd_std))
    print('''\n==>ave_asd_OC : {0}-{1}'''.format(total_asd_OC, total_asd_OC_std))
    print('''\n==>ave_asd_OD : {0}-{1}'''.format(total_asd_OD, total_asd_OD_std))
    print('''\n==>ave_average_asd : {0}-{1}'''.format(total_asd, total_asd_std))
    print('''\n==>ave_rec_OC : {0}-{1}'''.format(total_rec_OC, total_rec_OC_std))
    print('''\n==>ave_rec_OD : {0}-{1}'''.format(total_rec_OD, total_rec_OD_std))
    print('''\n==>ave_spec_OC : {0}-{1}'''.format(total_spec_OC, total_spec_OC_std))
    print('''\n==>ave_spec_OD : {0}-{1}'''.format(total_spec_OD, total_spec_OD_std))
    with open(osp.join(output_path, '../test' + str(data_params['datasettest'][0]) + '_log.csv'), 'a') as f:
        elapsed_time = (
                datetime.now(pytz.timezone('Asia/Hong_Kong')) -
                timestamp_start).total_seconds()

        log = [['batch-size: '] + [batch_size] + [eval_params['model_file']] + ['cup dice coefficence: '] + \
               [val_cup_dice]+['-']+[val_cup_dice_std] + ['disc dice coefficence: '] + \
               [val_disc_dice]+['-']+[val_disc_dice_std] + ['total dice coefficence: '] + \
               [total_dice]+['-']+[total_dice_std] + ['average_hd_OC: '] + \
               [total_hd_OC]+['-']+[total_hd_OC_std] + ['average_hd_OD: '] + \
               [total_hd_OD]+['-']+[total_hd_OD_std] + ['total_hd: '] + \
               [total_hd]+['-']+[total_hd_std] + ['ave_asd_OC: '] + \
               [total_asd_OC]+['-']+[total_asd_OC_std] + ['average_asd_OD: '] + \
               [total_asd_OD]+['-']+[total_asd_OD_std] + ['average_asd: '] + \
               [total_asd]+['-']+[total_asd_std] + ['elapse time: '] + \
               [elapsed_time]]
        log = map(str, log)
        f.write(','.join(log) + '\n')


if __name__ == '__main__':
    main()