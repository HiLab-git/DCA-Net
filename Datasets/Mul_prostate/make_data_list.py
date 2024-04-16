import os
import re
import random
import numpy as np

DATA_PATH = '../data/npz_data'
site_list = ['BIDMC', 'HK', 'I2CVB', 'ISBI', 'ISBI_1.5', 'UCL']
SAVE_PATH = './Mul_s'

def tryint(s):  # 将元素中的数字转换为int后再排序
    try:
        return int(s)
    except ValueError:
        return s

def str2int(v_str): # 将元素中的字符串和数字分割开
    return [tryint(sub_str) for sub_str in re.split('([0-9]+)', v_str)]

def make_filelist():
    for site in site_list:
        file_folder = os.path.join(DATA_PATH, site)
        case_folder = sorted([x for x in os.listdir(file_folder)])
        case_folder.sort(key=str2int)
        # case_folder_list = list(set(case_folder))
        # case_folder_list = sorted(case_folder_list)

        if not os.path.isdir(os.path.join(SAVE_PATH)):
            os.makedirs(os.path.join(SAVE_PATH))

        # train_list = [os.path.join(DATA_PATH.replace('..', '/mnt/lustre/guran'), site, x) for x in case_folder]
        train_list = [os.path.join(site, x.split('.')[0]) for x in case_folder]

        text_save(os.path.join(SAVE_PATH, site+'_train_list'), train_list)


def text_save(filename, data):      # filename: path to write CSV, data: data list to be written.
    file = open(filename, 'w+')
    for i in range(len(data)):
        s = str(data[i]).replace('[', '').replace(']', '')
        s = s.replace("'", '').replace(',', '') + '\n'
        file.write(s)
    file.close()
    print("Save {} successfully".format(filename.split('/')[-1]))


def text_lb_save(filename, data):      # filename: path to write CSV, data: data list to be written.
    file = open(filename, 'w+')
    for i in range(len(data)):
        s = str(data[i]).replace('[', '').replace(']', '')
        if i % 2 == 1:
            s = s.replace("'", '').replace('_segmentation.nii.gz,', '_segmentation.nii.gz') + '\n'
        else:
            s = s.replace("'", '') + ','
        file.write(s)
    file.close()
    print("Save {} successfully".format(filename.split('/')[-1]))

if __name__ == '__main__':
    make_filelist()
