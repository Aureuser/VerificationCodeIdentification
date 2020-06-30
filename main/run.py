# -*- coding: utf-8 -*- 
# @Time : 2020/5/13 15:04 
# @Author : lxd 
# @File : run.py
from torchvision import transforms
from utils.util import image_train_test_split
from utils.ImageDataset import ImageDataset
from torch.utils.data import DataLoader
from utils.train import train
from model.CNN_model import CNN_model

def main():
    transform = transforms.Compose(
        [
            # transforms.Resize(100),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    train_img, test_img = image_train_test_split(root='../data/Discuz', p=5/6)
    train_imageset = ImageDataset(root='../data/Discuz/', imgs=train_img, transform=transform)
    train_img_loader = DataLoader(train_imageset, batch_size=128, shuffle=True, pin_memory=True)
    test_imageset = ImageDataset(root='../data/Discuz/', imgs=test_img, transform=transform)
    test_img_loader = DataLoader(test_imageset, batch_size=128, shuffle=False, pin_memory=True)
    model = CNN_model()
    train(model=model, train_loader=train_img_loader, test_loader=test_img_loader, step=128,
          epochs=1024, lr=0.001, use_cuda=True)

if __name__ == '__main__':
    main()