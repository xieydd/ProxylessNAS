'''
@Description: prepare for image
@Author: xieydd
@Date: 2019-08-13 15:52:14
@LastEditTime: 2019-08-22 14:06:39
@LastEditors: Please set LastEditors
'''
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as dset
import os

'''
@description: Cutout Tec to improved Regularzation via random section of input images during train \
              , you can find description in https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
'''
class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask

        return img


def data_transforms(dataset, cutout_length):
    dataset = dataset.lower()
    if dataset == 'cifar10':
        MEAN = [0.49139968, 0.48215827, 0.44653124]
        STD = [0.24703233, 0.24348505, 0.26158768]
        transf = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()
        ]
    elif dataset == 'mnist':
        MEAN = [0.13066051707548254]
        STD = [0.30810780244715075]
        transf = [
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1)
        ]
    elif dataset == 'fashionmnist':
        MEAN = [0.28604063146254594]
        STD = [0.35302426207299326]
        transf = [
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1),
            transforms.RandomVerticalFlip()
        ]
    else:
        raise ValueError('not expected dataset = {}'.format(dataset))

    normalize = [
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ]

    train_transform = transforms.Compose(transf + normalize)
    valid_transform = transforms.Compose(normalize)

    if cutout_length > 0:
        train_transform.transforms.append(Cutout(cutout_length))

    return train_transform, valid_transform

def imagenet_transform(data_path, batch_size, workers):
    traindir = os.path.join(data_path, 'train')
    validdir = os.path.join(data_path, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_data = dset.ImageFolder(
        traindir,
        transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.2),
        transforms.ToTensor(),
        normalize,
        ]))
    valid_data = dset.ImageFolder(
        validdir,
        transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=workers)

    valid_loader = torch.utils.data.DataLoader(
        valid_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=workers)
    return train_loader, valid_loader
  