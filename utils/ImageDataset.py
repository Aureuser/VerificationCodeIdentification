# -*- coding: utf-8 -*- 
# @Time : 2020/5/14 10:09 
# @Author : lxd 
# @File : ImageDataset.py

from torchvision import transforms, datasets as ds
import torchvision as tv
from torch.utils.data import DataLoader, Dataset
import os, torch
from PIL import Image
import random
from utils.util import image_train_test_split


class ImageDataset(Dataset):
    def __init__(self, root, imgs, transform=None):
        super(ImageDataset, self).__init__()
        self.root = root
        self.imgs = imgs
        self.transform = transform

    def __len__(self):
        return len(self.imgs)


    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.imgs[idx])
        img = Image.open(img_path).convert('RGB')
        char_label = self.imgs[idx].split('.', maxsplit=2)[0]
        label = self._get_label(char_label)
        label = torch.Tensor(label)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def _get_label(self, label):
        if len(label) != 4:
            raise Exception
        labels = []
        for char in label:
            char_label = [0] * 62
            if ord('0') <= ord(char) <= ord('9'):
                char_label[ord(char)-ord('0')] = 1
            elif ord('a') <= ord(char) <= ord('z'):
                char_label[ord(char)-ord('a') + 10] = 1
            elif ord('A') <= ord(char) <= ord('Z'):
                char_label[ord(char)-ord('a') + 10 + 26] = 1
            else:
                raise Exception
            labels.extend(char_label)
        return labels

if __name__ == '__main__':
    transform =transforms.Compose(
        [
            # transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    train_img, test_img = image_train_test_split(root='../data/Discuz', p=5/6)
    train_imageset = ImageDataset(root='../data/Discuz/', imgs=train_img, transform=transform)
    train_img_loader = DataLoader(train_imageset, batch_size=4)
    for i, (img, label) in enumerate(train_img_loader):
        print(img.size())