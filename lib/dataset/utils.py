import cv2
import random
import numpy as np
from sklearn.metrics import confusion_matrix

import torch

label_colours = [(128, 64, 128), (244, 35, 231), (69, 69, 69), (102, 102, 156), (190, 153, 153), (153, 153, 153),
                 (250, 170, 29), (219, 219, 0), (106, 142, 35), (152, 250, 152), (69, 129, 180), (219, 19, 60),
                 (255, 0, 0), (0, 0, 142), (0, 0, 69), (0, 60, 100), (0, 79, 100), (0, 0, 230), (119, 10, 32),
                 (0, 0, 0)]
label_colours_camvid = [(128, 128, 128), (128, 0, 0), (192, 192, 128), (128, 64, 128), (0, 0, 192), (128, 128, 0),
                        (192, 128, 128), (64, 64, 128), (64, 0, 128), (64, 64, 0), (0, 128, 192), (0, 0, 0)]


def transform_im(img):
    img = img[:, :, ::-1]
    img = img.transpose((2, 0, 1))
    img = img.astype(np.float32) / 255.0
    return img


def randomcrop(img_list, crop_size=(512, 1024)):
    _, h, w = img_list[0].shape
    crop_h, crop_w = crop_size
    top = np.random.randint(0, h - crop_h + 1)
    left = np.random.randint(0, w - crop_w + 1)

    for i in range(len(img_list)):
        if len(img_list[i].shape) == 3:
            img_list[i] = img_list[i][:, top:top + crop_h, left:left + crop_w]
        else:
            img_list[i] = img_list[i][top:top + crop_h, left:left + crop_w]

    return img_list


def decode_labels(mask):
    h, w = mask.shape
    mask[mask == 255] = 19
    color_table = np.array(label_colours, dtype=np.float32)
    out = np.take(color_table, mask, axis=0)
    out = out.astype(np.uint8)
    out = out[:, :, ::-1]
    return out


def decode_labels_camvid(mask):
    h, w = mask.shape
    mask[mask == 255] = 11
    color_table = np.array(label_colours_camvid, dtype=np.float32)
    out = np.take(color_table, mask, axis=0)
    out = out.astype(np.uint8)
    out = out[:, :, ::-1]
    return out


class runningScore(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask],
            minlength=n_class**2,
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self, return_class=False):
        hist = self.confusion_matrix
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)

        if return_class:
            return mean_iu, iu
        else:
            return mean_iu

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
