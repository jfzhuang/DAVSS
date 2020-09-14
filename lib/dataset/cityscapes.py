import os
import sys
import cv2
import random
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from lib.dataset.utils import transform_im, randomcrop


class cityscapes_video_dataset(Dataset):
    def __init__(self, data_path, gt_path, list_path, crop_size=(512, 1024)):
        self.data_path = data_path
        self.gt_path = gt_path
        self.get_list(list_path)
        self.crop_size = crop_size

    def __len__(self):
        return len(self.gt_label_name)

    def __getitem__(self, idx):
        img_1 = cv2.imread(os.path.join(self.data_path, self.img_1_name[idx]))
        img_1 = transform_im(img_1)
        img_2 = cv2.imread(os.path.join(self.data_path, self.img_2_name[idx]))
        img_2 = transform_im(img_2)
        img_3 = cv2.imread(os.path.join(self.data_path, self.img_3_name[idx]))
        img_3 = transform_im(img_3)

        gt_label = cv2.imread(os.path.join(self.gt_path, self.gt_label_name[idx]), 0)

        if self.crop_size is not None:
            [img_1, img_2, img_3, gt_label] = randomcrop([img_1, img_2, img_3, gt_label], crop_size=self.crop_size)

        img_1 = torch.from_numpy(img_1)
        img_2 = torch.from_numpy(img_2)
        img_3 = torch.from_numpy(img_3)
        gt_label = torch.from_numpy(gt_label.astype(np.int64))

        return [img_1, img_2, img_3], gt_label

    def get_list(self, list_path):
        self.img_1_name = []
        self.img_2_name = []
        self.img_3_name = []
        self.gt_label_name = []

        with open(list_path, 'r') as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            img_3_name, gt_label_name = line.split()
            img_3_id = int(img_3_name[-22:-16])

            for j in range(2, 11):
                img_1_id = img_3_id - j
                img_2_id = img_3_id - random.randint(1, j - 1)
                img_1_name = img_3_name.replace('{:06d}_leftImg8bit.png'.format(img_3_id),
                                                '{:06d}_leftImg8bit.png'.format(img_1_id))
                img_2_name = img_3_name.replace('{:06d}_leftImg8bit.png'.format(img_3_id),
                                                '{:06d}_leftImg8bit.png'.format(img_2_id))

                self.img_1_name.append(img_1_name)
                self.img_2_name.append(img_2_name)
                self.img_3_name.append(img_3_name)
                self.gt_label_name.append(gt_label_name)


class cityscapes_video_dataset_PDA(Dataset):
    def __init__(self, data_path, gt_path, list_path):
        self.data_path = data_path
        self.gt_path = gt_path
        self.get_list(list_path)

    def __len__(self):
        return len(self.gt_label_name)

    def __getitem__(self, idx):
        img_list = []

        for name in self.img_name[idx]:
            img = cv2.imread(os.path.join(self.data_path, name))
            img = transform_im(img)
            img = torch.from_numpy(img)
            img_list.append(img)

        gt_label = cv2.imread(os.path.join(self.gt_path, self.gt_label_name[idx]), 0)
        gt_label = torch.from_numpy(gt_label.astype(np.int64))

        return img_list, gt_label

    def get_list(self, list_path):
        self.img_name = []
        self.gt_label_name = []

        with open(list_path, 'r') as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            name, gt_label_name = line.split()
            self.gt_label_name.append(gt_label_name)
            img_id = int(name[-22:-16])
            img_name_list = []
            for j in range(10):
                img_name = name.replace('{:06d}_leftImg8bit.png'.format(img_id),
                                        '{:06d}_leftImg8bit.png'.format(img_id - 9 + j))
                img_name_list.append(img_name)
            self.img_name.append(img_name_list)


if __name__ == '__main__':

    ###############################################################################################################################
    data_path = '/gpub'
    mask_path = '/gdata1/zhuangjf/DAVSS/result1/deeplab_results'
    list_path = '/ghome/zhuangjf/git_repo/DAVSS/data/list/cityscapes/train.txt'
    gt_path = '/gdata/zhuangjf/cityscapes/original'
    batch_size = 1
    shuffle = False

    train_data = cityscapes_video_dataset(data_path, gt_path, mask_path, list_path)
    data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=shuffle, num_workers=4)

    for i, data in enumerate(data_loader):
        img_list, img_mask_list, gt_label = data
        print('{}/{}'.format(i, len(data_loader)), len(img_list), img_list[0].shape, len(img_mask_list),
              img_mask_list[0].shape, gt_label.shape)

    # ###############################################################################################################################
    # data_path = '/gpub'
    # list_path = '/ghome/zhuangjf/git_repo/DAVSS/data/list/cityscapes/val.txt'
    # gt_path = '/gdata/zhuangjf/cityscapes/original'
    # batch_size = 1
    # shuffle = False

    # val_data = cityscapes_video_dataset_PDA(data_path, gt_path, list_path)
    # data_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=shuffle, num_workers=4)

    # for i, data in enumerate(data_loader):
    #     img_list, gt_label = data
    #     print('{}/{}'.format(i, len(data_loader)), len(img_list), img_list[0].shape, gt_label.shape)
